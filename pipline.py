"""
Copyright (C) 2025 Fu Tszkok

:module: pipline
:function: Orchestrates the entire Machine Translation workflow, including data preparation, training, and evaluation.
:author: Fu Tszkok
:date: 2025-12-25
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os
import time
import argparse
import warnings
from datetime import timedelta

import torch
import yaml
from tqdm import trange

from dataloader import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from evaluate import EvalEngine
from network.lstm import build_model as build_rnn
from network.transformer import build_model as build_transformer
from train import TrainingEngine
from utils import (TrainingLogger, EarlyStopper, plot_metrics, plot_attention_heatmap, load_pretrained_vectors, save_results_to_json)

warnings.filterwarnings("ignore", category=UserWarning)


def find_chinese_font():
    """Attempts to locate a Chinese font file on the system for matplotlib.
    :return: Path to the font file if found, otherwise None.
    """
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    ]
    for path in font_paths:
        if os.path.exists(path):
            print(f"[Info] Found Chinese font at: {path}")
            return path
    print("[Warning] No default Chinese font found. Attention map may display squares for Chinese characters.")
    return None


def load_config(path):
    """Loads configuration from a YAML file.
    :param path: Path to the config file.
    :return: Dictionary containing configuration parameters.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def init_model_and_data(cfg, device, mode='train'):
    """Initializes the datasets, vocabulary, and model architecture.
    :param cfg: Configuration dictionary.
    :param device: Computing device.
    :param mode: 'train' or 'test' (determines embedding loading).
    :return: Tuple (model, train_loader, val_loader, test_loader, src_vocab, tgt_vocab).
    """
    print("Loading datasets and vocabularies...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(cfg['data'])

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(f"Source Vocab Size: {src_vocab_size}, Target Vocab Size: {tgt_vocab_size}")

    model_config = cfg['model']
    model_config['src_pad_idx'] = PAD_IDX
    model_config['tgt_pad_idx'] = PAD_IDX
    model_name = model_config['name']

    # Build model based on configuration type
    if 'lstm' in model_name:
        model_config['enc_emb_dim'], model_config['dec_emb_dim'] = cfg['model']['emb_size'], cfg['model']['emb_size']
        model_config['hid_dim'] = cfg['model']['hidden_size']
        model_config['rnn_type'] = model_name
        model_config['attn_method'] = cfg['model'].get('attn_method', 'additive')
        model = build_rnn(model_config, src_vocab_size, tgt_vocab_size, device)
    elif 'transformer' in model_name:
        model_config['transformer_type'] = 'optimal' if 'optimal' in model_name else 'standard'
        model = build_transformer(model_config, src_vocab_size, tgt_vocab_size, device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Load pre-trained embeddings only during training initialization
    if mode == 'train':
        _load_embeddings(cfg, model, src_vocab, tgt_vocab)

    return model, train_loader, val_loader, test_loader, src_vocab, tgt_vocab


def _load_embeddings(cfg, model, src_vocab, tgt_vocab):
    """Helper function to load pre-trained word embeddings into the model.
    :param cfg: Configuration dictionary.
    :param model: The initialized model.
    :param src_vocab: Source vocabulary.
    :param tgt_vocab: Target vocabulary.
    """
    src_emb_path = cfg['data'].get('pretrained_src_emb', '')
    if src_emb_path and os.path.exists(src_emb_path):
        print(f"[Info] Loading Source Embeddings from {src_emb_path} ...")
        embedding_layer = None
        # Locate embedding layer in different model architectures
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'embedding'):
            embedding_layer = model.encoder.embedding
        elif hasattr(model, 'embedding'):
            embedding_layer = model.embedding

        if embedding_layer is not None:
            weights = load_pretrained_vectors(src_vocab, src_emb_path, cfg['model']['emb_size'])
            if weights is not None:
                embedding_layer.weight.data.copy_(weights)
                embedding_layer.weight.requires_grad = True

    tgt_emb_path = cfg['data'].get('pretrained_tgt_emb', '')
    if tgt_emb_path and os.path.exists(tgt_emb_path):
        print(f"[Info] Loading Target Embeddings from {tgt_emb_path} ...")
        embedding_layer = None
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'embedding'):
            embedding_layer = model.decoder.embedding

        if embedding_layer is not None:
            weights = load_pretrained_vectors(tgt_vocab, tgt_emb_path, cfg['model']['emb_size'])
            if weights is not None:
                embedding_layer.weight.data.copy_(weights)
                embedding_layer.weight.requires_grad = True


def train_pipeline(cfg):
    """Executes the complete training pipeline.
    :param cfg: Configuration dictionary.
    """
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    model_name = cfg['model']['name']

    print(f"=== Starting Training Pipeline for {model_name} ===")
    print(f"Device: {device}")

    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    model, train_loader, val_loader, _, src_vocab, tgt_vocab = init_model_and_data(cfg, device, mode='train')

    tgt_vocab_size = len(tgt_vocab)

    # Initialize Engine components
    trainer = TrainingEngine(model, device, cfg, tgt_vocab_size, PAD_IDX)
    evaluator = EvalEngine(model, device, cfg, src_vocab, tgt_vocab, PAD_IDX, SOS_IDX, EOS_IDX)
    logger = TrainingLogger(cfg['log_dir'], model_name)

    # Initialize Early Stopping
    es_patience = cfg['train']['early_stopping']['patience']
    es_mode = cfg['train']['early_stopping'].get('mode', 'max')
    es_metric = cfg['train']['early_stopping'].get('metric', 'bleu')
    early_stopper = EarlyStopper(patience=es_patience, mode=es_mode)

    print(f"Start Training... (Early Stopping on {es_metric}, mode={es_mode})")

    for epoch in range(cfg['train']['epochs']):
        # Train
        t_loss = trainer.train_one_epoch(train_loader, epoch + 1)

        # Validate
        v_loss = evaluator.evaluate_loss(val_loader, trainer.criterion)
        # Generate translations for metric calculation (using greedy for speed during validation)
        _, _, _, _, _, val_bleu, val_rouge, _ = evaluator.generate_translations(val_loader, use_beam_search=False)

        # Log and step scheduler
        logger.add_entry(epoch + 1, t_loss, v_loss, val_bleu, val_rouge)
        trainer.step_scheduler(v_loss)

        print(f"Epoch {epoch + 1} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val BLEU: {val_bleu:.2f}")

        # Check Early Stopping
        current_score = v_loss if es_metric == 'loss' else val_bleu
        early_stopper(current_score)

        if early_stopper.is_new_best:
            save_path = os.path.join(cfg['save_dir'], "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f">>> Found new best model ({es_metric}={current_score:.4f}). Saved to {save_path}")

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    metrics_df = logger.save_to_csv()
    plot_metrics(metrics_df, model_name, cfg['log_dir'])
    print("=== Training Finished ===\n")


def test_pipeline(cfg):
    """Executes the evaluation pipeline using the best trained model.
    :param cfg: Configuration dictionary.
    """
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    model_name = cfg['model']['name']

    heatmap_dir = os.path.join(cfg['log_dir'], "attention_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)

    print(f"=== Starting Evaluation Pipeline for {model_name} ===")
    print(f"Device: {device}")

    model, _, _, test_loader, src_vocab, tgt_vocab = init_model_and_data(cfg, device, mode='test')

    # Load best checkpoint
    best_model_path = os.path.join(cfg['save_dir'], "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}. Please train first.")

    print(f"Loading checkpoint from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    evaluator = EvalEngine(model, device, cfg, src_vocab, tgt_vocab, PAD_IDX, SOS_IDX, EOS_IDX)
    chinese_font_path = find_chinese_font()

    print("Running Inference on Test Set (with Beam Search if applicable)...")
    start_time = time.time()

    # Run inference with detailed metrics and Beam Search
    test_srcs, test_hyps, test_refs, test_attns, time_stats, corpus_bleu, corpus_rouge, sent_metrics = \
        evaluator.generate_translations(test_loader, calculate_sentence_metrics=True, use_beam_search=True)

    inference_time = time.time() - start_time
    print(f"Inference finished in {str(timedelta(seconds=int(inference_time)))}")

    # Save translation results
    json_path = os.path.join(cfg['log_dir'], "translations.json")
    save_results_to_json(test_srcs, test_hyps, test_refs, json_path)

    print("\nGenerating Attention Heatmaps...")
    for i in trange(len(test_srcs)):
        try:
            src_tokens = test_loader.dataset.data[i]['zh_tokens']
            hyp_tokens = test_hyps[i].split()
            attn_matrix = test_attns[i]

            if isinstance(attn_matrix, torch.Tensor):
                attn_matrix = attn_matrix.numpy()

            # Handle multi-head attention averaging
            if attn_matrix.ndim == 3:
                attn_matrix = attn_matrix.mean(axis=0)

            # Slice attention matrix to valid lengths
            valid_src_len = min(len(src_tokens), attn_matrix.shape[1])
            valid_tgt_len = min(len(hyp_tokens), attn_matrix.shape[0])

            attn_matrix_sliced = attn_matrix[:valid_tgt_len, :valid_src_len]

            current_src_tokens = src_tokens[:valid_src_len]
            current_hyp_tokens = hyp_tokens[:valid_tgt_len]

            save_path = os.path.join(heatmap_dir, f"attention_example_{i}.png")
            plot_attention_heatmap(attn_matrix_sliced, current_src_tokens, current_hyp_tokens, save_path, font_path=chinese_font_path)
        except Exception as e:
            print(f"Could not generate attention plot for example {i}: {e}")

    # Print final report
    print("\n" + "=" * 96)
    print(" " * 38 + "EVALUATION REPORT")
    print("=" * 96)

    header = f"| {'Model':<25} | {'Corpus BLEU':<15} | {'Corpus ROUGE-L':<18} | {'Inference Time (ms)':<25} |"
    separator = f"|{'-' * 27}|{'-' * 17}|{'-' * 20}|{'-' * 27}|"
    row1 = f"| {model_name:<25} | {corpus_bleu:<15.2f} | {corpus_rouge:<18.2f} | {f'{time_stats[0] * 1000:.2f} ± {time_stats[1] * 1000:.2f}':<25} |"

    bleu_mean, bleu_std = sent_metrics.get('bleu_mean_std', (0, 0))
    rouge_mean, rouge_std = sent_metrics.get('rouge_mean_std', (0, 0))
    header2 = f"| {'Metric (Sentence Level)':<43} | {'Mean ± Std':<46} |"
    separator2 = f"|{'-' * 45}|{'-' * 48}|"
    row2 = f"| {'Sentence BLEU':<43} | {f'{bleu_mean:.2f} ± {bleu_std:.2f}':<46} |"
    row3 = f"| {'Sentence ROUGE-L':<43} | {f'{rouge_mean:.2f} ± {rouge_std:.2f}':<46} |"

    print(separator)
    print(header)
    print(separator)
    print(row1)
    print(separator2)
    print(header2)
    print(separator2)
    print(row2)
    print(row3)
    print(separator2)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Machine Translation Pipeline")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'evaluate', 'all'],
                        help='Pipeline mode: train (only training), test (only inference), all (train then test)')

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == 'train':
        train_pipeline(config)
    elif args.mode == 'evaluate':
        test_pipeline(config)
    elif args.mode == 'all':
        train_pipeline(config)
        test_pipeline(config)
