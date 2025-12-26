"""
Copyright (C) 2025 Fu Tszkok

:module: utils
:function: Provides utility classes and functions for training monitoring, logging, visualization, and result management.
:author: Fu Tszkok
:date: 2025-12-21
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class EarlyStopper:
    """Implements Early Stopping to terminate training when validation metric stops improving."""

    def __init__(self, patience=3, mode="min", delta=0.0):
        """Initializes the EarlyStopper.
        :param patience: Number of epochs to wait for improvement before stopping.
        :param mode: 'min' for loss, 'max' for metrics like BLEU.
        :param delta: Minimum change required to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_new_best = False

    def __call__(self, score):
        """Updates the early stopping state based on the current score.
        :param score: The current validation metric (e.g., loss or BLEU).
        """
        self.is_new_best = False
        if self.best_score is None:
            self.best_score = score
            self.is_new_best = True
        elif self.mode == "min":
            # Check if loss decreased
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
                self.is_new_best = True
            else:
                self.counter += 1
        elif self.mode == "max":
            # Check if metric increased
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
                self.is_new_best = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


class TrainingLogger:
    """Logs training history and saves it to CSV."""

    def __init__(self, log_dir, model_name):
        """Initializes the Logger.
        :param log_dir: Directory to save logs.
        :param model_name: Name of the model.
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.history = []

    def add_entry(self, epoch, train_loss, val_loss, val_bleu, val_rouge):
        """Adds a record for a single epoch."""
        self.history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_bleu': val_bleu, 'val_rouge': val_rouge})

    def save_to_csv(self):
        """Saves the recorded history to a CSV file."""
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.log_dir, f"{self.model_name}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Training metrics saved to {csv_path}")
        return df


def load_pretrained_vectors(vocab, vector_path, emb_dim):
    """Loads pre-trained word vectors (e.g., GloVe, Word2Vec) into the embedding matrix.
    :param vocab: Vocabulary object containing the mapping from token to ID.
    :param vector_path: Path to the pre-trained vector file.
    :param emb_dim: Dimension of the embeddings.
    :return: Tensor containing the weight matrix.
    """
    if not os.path.exists(vector_path):
        print(f"Warning: Vector file not found at {vector_path}")
        return None

    embeddings = {}
    print(f"Loading vectors from {vector_path}...")

    with open(vector_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Check header line for format (word2vec often has count/dim on first line)
        first_line = f.readline().rstrip().split()
        if len(first_line) == 2:
            pass
        else:
            f.seek(0)

        for line in f:
            parts = line.rstrip().split()
            # Skip malformed lines
            if len(parts) < emb_dim + 1:
                continue
            vector_parts = parts[-emb_dim:]
            word = " ".join(parts[:-emb_dim])
            try:
                vector = np.array(vector_parts, dtype='float32')
                if len(vector) == emb_dim:
                    embeddings[word] = vector
            except ValueError:
                continue

    # Initialize weights matrix
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    hits = 0
    scale = 1.0 / np.sqrt(emb_dim)

    # Fill matrix with pre-trained vectors or random initialization
    for idx, word in vocab.id2token.items():
        if idx == 0:  # Skip PAD
            continue
        if word in embeddings:
            weights_matrix[idx] = embeddings[word]
            hits += 1
        else:
            # Random init for OOV words
            weights_matrix[idx] = np.random.normal(scale=scale, size=(emb_dim,))
    print(f"Pretrained vectors hit rate: {hits} / {matrix_len} ({hits / matrix_len:.2%})")
    return torch.tensor(weights_matrix, dtype=torch.float)


def plot_metrics(df, model_name, log_dir):
    """Generates and saves plots for Loss and Metrics (BLEU/ROUGE).
    :param df: Pandas DataFrame containing training history.
    :param model_name: Name of the model.
    :param log_dir: Directory to save plots.
    """
    sns.set_theme(style="whitegrid")

    # Plot Loss Curve
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss')
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Validation Loss', ax=ax)
    ax.set_title(f'{model_name} Loss Curve', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{model_name}_loss.png"))
    plt.close()

    # Plot Metric Curve (Double Y-Axis for BLEU and ROUGE)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(data=df, x='epoch', y='val_bleu', label='Validation BLEU', ax=ax1, color='b', marker='o')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='epoch', y='val_rouge', label='Validation ROUGE', ax=ax2, color='r', marker='s')
    ax2.set_ylabel('ROUGE-L Score', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'{model_name} Validation Metrics', fontsize=16)
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(log_dir, f"{model_name}_metrics.png"))
    plt.close()


def plot_attention_heatmap(attention, sentence, predicted_sentence, save_path, font_path=None):
    """Visualizes the Attention weights as a heatmap.
    :param attention: Attention matrix [tgt_len, src_len].
    :param sentence: List of source tokens.
    :param predicted_sentence: List of target tokens.
    :param save_path: Path to save the image.
    :param font_path: Optional path to a font file (useful for CJK characters).
    """
    if font_path and os.path.exists(font_path):
        import matplotlib.font_manager as fm
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False

    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().detach().numpy()

    # Average attention heads if multi-head attention is provided
    if attention.ndim == 3:
        attention = attention.mean(axis=0)

    src_len = len(sentence)
    tgt_len = len(predicted_sentence)
    figsize = (max(6, src_len * 0.8), max(6, tgt_len * 0.8))

    plt.figure(figsize=figsize)
    ax = sns.heatmap(attention, xticklabels=sentence, yticklabels=predicted_sentence, cmap="Blues", linewidths=0.5,
                     linecolor='gray', square=True, cbar_kws={"shrink": 0.8})

    ax.xaxis.tick_top()
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.title("Attention Alignment", y=1.1, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results_to_json(sources, hypotheses, references, save_path):
    """Saves the translation results to a JSON file.
    :param sources: List of source sentences.
    :param hypotheses: List of generated translations.
    :param references: List of reference translations.
    :param save_path: Output file path.
    """
    results = []
    for src, hyp, ref in zip(sources, hypotheses, references):
        results.append({"src": src, "ref": ref, "hyp": hyp})
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {save_path}")
