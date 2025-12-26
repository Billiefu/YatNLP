"""
Copyright (C) 2025 Fu Tszkok

:module: evaluate
:function: Provides evaluation engines for RNN and Transformer models, including Greedy and Beam Search decoding, and metric calculation.
:author: Fu Tszkok
:date: 2025-12-25
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric import BleuCalculator, RougeCalculator


class EvalEngine:
    """Handles model evaluation, text generation, and metric calculation."""

    def __init__(self, model, device, config, src_vocab, tgt_vocab, pad_idx, sos_idx, eos_idx):
        """Initializes the Evaluation Engine.
        :param model: The trained Seq2Seq model.
        :param device: Computing device.
        :param config: Configuration dictionary.
        :param src_vocab: Source vocabulary object.
        :param tgt_vocab: Target vocabulary object.
        :param pad_idx: Padding token index.
        :param sos_idx: Start-of-sentence token index.
        :param eos_idx: End-of-sentence token index.
        """
        self.model = model
        self.device = device
        self.config = config
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = config['data'].get('tgt_max_len', 50)
        self.model_name = config['model']['name']

        # Initialize metric calculators
        self.bleu_calc = BleuCalculator()
        self.rouge_calc = RougeCalculator()

    def greedy_decode_rnn(self, src):
        """Performs Greedy Decoding for RNN-based models.
        :param src: Source sequence tensor [batch_size, src_len].
        :return: Tuple (decoded_ids, attentions).
        """
        batch_size = src.shape[0]
        encoder_outputs, hidden, cell = self.model.encoder(src)
        inputs = torch.tensor([self.sos_idx] * batch_size, device=self.device)

        decoded_ids = []
        attentions = []

        # Keep track of generated tokens to apply repetition penalty
        generated_tokens = torch.zeros(batch_size, self.max_len, dtype=torch.long, device=self.device)

        for t in range(self.max_len):
            output, hidden, cell, attn_weights = self.model.decoder(inputs, hidden, cell, encoder_outputs)
            attentions.append(attn_weights)

            # Apply repetition penalty for the immediate context window
            if t > 0:
                penalty_value = -float('inf')
                start_window = max(0, t - 1)
                for k in range(start_window, t):
                    past_toks = generated_tokens[:, k]
                    # Penalize previously generated tokens by setting their logits to -inf
                    output.scatter_(1, past_toks.unsqueeze(1), penalty_value)

            # Greedy selection: Choose token with highest probability
            top1 = output.argmax(1)
            decoded_ids.append(top1)

            generated_tokens[:, t] = top1
            inputs = top1

            # Stop if batch_size is 1 and EOS is generated
            if batch_size == 1 and top1.item() == self.eos_idx:
                break

        decoded_ids = torch.stack(decoded_ids, dim=1)
        attentions = torch.stack(attentions, dim=1)
        return decoded_ids, attentions

    def beam_decode_rnn(self, src, beam_size=5, alpha=0.7):
        """Performs Beam Search Decoding for RNN-based models.
        :param src: Source sequence tensor [1, src_len].
        :param beam_size: The width of the beam.
        :param alpha: Length normalization penalty factor.
        :return: Tuple (final_ids, final_attn).
        """
        assert src.shape[0] == 1, "Beam search currently supports batch_size=1 only."

        # Encode and expand encoder outputs for beam processing
        encoder_outputs, hidden, cell = self.model.encoder(src)
        encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)

        # Handle hidden state dimensions (depending on stacked vs single layer)
        if hidden.dim() == 2:
            hidden = hidden.expand(beam_size, -1).contiguous()
            cell = cell.expand(beam_size, -1).contiguous()
            is_hidden_3d = False
        else:
            hidden = hidden.expand(-1, beam_size, -1).contiguous()
            cell = cell.expand(-1, beam_size, -1).contiguous()
            is_hidden_3d = True

        # Initialize beam variables
        decoder_input = torch.tensor([self.sos_idx] * beam_size, device=self.device)
        beam_scores = torch.zeros(beam_size, device=self.device)
        beam_scores[1:] = -float('inf')  # Only the first beam path is valid initially
        sequences = [[self.sos_idx] for _ in range(beam_size)]
        finished_sequences = []

        # Store attention maps for visualization
        beam_attentions = torch.zeros(beam_size, self.max_len, src.shape[1], device=self.device)

        for t in range(self.max_len):
            output, hidden, cell, attn_weights = self.model.decoder(decoder_input, hidden, cell, encoder_outputs)

            beam_attentions[:, t, :] = attn_weights
            log_probs = F.log_softmax(output, dim=1)

            # Calculate cumulative scores: previous_score + current_log_prob
            next_scores = beam_scores.unsqueeze(1) + log_probs
            flat_scores = next_scores.view(-1)

            # Select top-k best paths
            best_scores, best_indices = flat_scores.topk(beam_size, sorted=True)

            # Map flat indices back to beam index and token index
            prev_beam_indices = torch.div(best_indices, self.tgt_vocab.token2id.__len__(), rounding_mode='floor')
            token_indices = best_indices % self.tgt_vocab.token2id.__len__()

            new_sequences = []
            new_scores = []

            # Prepare tensors for next step
            if is_hidden_3d:
                new_hidden = torch.zeros_like(hidden)
                new_cell = torch.zeros_like(cell)
            else:
                new_hidden = torch.zeros_like(hidden)
                new_cell = torch.zeros_like(cell)

            new_attentions = torch.zeros_like(beam_attentions)
            active_beams_count = 0

            # Construct new beams
            for i in range(beam_size):
                prev_idx = prev_beam_indices[i].item()
                token_idx = token_indices[i].item()
                score = best_scores[i].item()

                new_seq = sequences[prev_idx] + [token_idx]

                if token_idx == self.eos_idx:
                    # Sequence finished: apply length normalization and store
                    final_len = len(new_seq) - 1
                    normalized_score = score / (final_len ** alpha)
                    attn_history = beam_attentions[prev_idx].clone()
                    finished_sequences.append((normalized_score, new_seq, attn_history))
                    new_scores.append(-float('inf'))  # Mark this beam as finished/invalid for next step
                    new_sequences.append(new_seq)
                else:
                    new_scores.append(score)
                    new_sequences.append(new_seq)

                    # Update hidden states for the surviving beams
                    if is_hidden_3d:
                        new_hidden[:, i, :] = hidden[:, prev_idx, :]
                        new_cell[:, i, :] = cell[:, prev_idx, :]
                    else:
                        new_hidden[i, :] = hidden[prev_idx, :]
                        new_cell[i, :] = cell[prev_idx, :]

                    new_attentions[i] = beam_attentions[prev_idx]
                    active_beams_count += 1

            beam_scores = torch.tensor(new_scores, device=self.device)
            sequences = new_sequences
            hidden = new_hidden
            cell = new_cell
            beam_attentions = new_attentions
            decoder_input = torch.tensor(token_indices.tolist(), device=self.device)

            if active_beams_count == 0:
                break

        # If no sequence finished naturally, choose the best current one
        if len(finished_sequences) == 0:
            best_idx = beam_scores.argmax().item()
            best_seq = sequences[best_idx]
            best_attn = beam_attentions[best_idx]
            return torch.tensor(best_seq[1:]).unsqueeze(0), best_attn.unsqueeze(0)

        # Select the best completed sequence
        finished_sequences.sort(key=lambda x: x[0], reverse=True)
        best_result = finished_sequences[0]

        final_ids = torch.tensor(best_result[1][1:]).unsqueeze(0)
        final_attn = best_result[2].unsqueeze(0)

        return final_ids, final_attn

    def greedy_decode_transformer(self, src):
        """Performs Greedy Decoding for Transformer models.
        :param src: Source sequence tensor [batch_size, src_len].
        :return: Tuple (decoded_ids, attentions).
        """
        batch_size = src.shape[0]
        src_mask = self.model.make_src_mask(src)
        enc_src = self.model.encoder(src, src_mask)
        trg_indexes = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
        attentions = None

        # Autoregressive generation
        for _ in range(self.max_len):
            tgt_mask = self.model.make_tgt_mask(trg_indexes)
            output, attentions = self.model.decoder(trg_indexes, enc_src, tgt_mask, src_mask)
            pred_token_logits = output

            # Select the token with the highest probability for the next position
            next_token = pred_token_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            trg_indexes = torch.cat([trg_indexes, next_token], dim=1)

            if batch_size == 1 and next_token.item() == self.eos_idx:
                break

        # Reshape attentions for return
        num_heads = self.config['model']['nhead']
        tgt_len = trg_indexes.shape[1] - 1
        src_len = src.shape[1]

        if attentions is not None:
            final_attentions = attentions.view(batch_size, num_heads, tgt_len, src_len)
        else:
            final_attentions = torch.zeros(batch_size, num_heads, tgt_len, src_len)

        return trg_indexes[:, 1:], final_attentions

    def beam_decode_transformer(self, src, beam_size=5, alpha=0.7):
        """Performs Beam Search Decoding for Transformer models.
        :param src: Source sequence tensor [1, src_len].
        :param beam_size: Beam width.
        :param alpha: Length normalization penalty factor.
        :return: Tuple (final_ids, final_attn).
        """
        assert src.shape[0] == 1, "Beam search currently supports batch_size=1 only."

        # Prepare encoder output and masks, expanded for beam size
        src_mask = self.model.make_src_mask(src)
        memory = self.model.encoder(src, src_mask)
        memory = memory.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)

        # Initial decoder input
        ys = torch.ones(beam_size, 1).fill_(self.sos_idx).type_as(src.data)

        beam_scores = torch.zeros(beam_size, device=self.device)
        beam_scores[1:] = -float('inf')

        finished_sequences = []

        for t in range(self.max_len):
            tgt_mask = self.model.make_tgt_mask(ys)
            out, _ = self.model.decoder(ys, memory, tgt_mask, src_mask)

            # Get logits for the last token
            logits = out[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            # Calculate top-k scores
            next_scores = beam_scores.unsqueeze(1) + log_probs
            flat_scores = next_scores.view(-1)
            best_scores, best_indices = flat_scores.topk(beam_size, sorted=True)

            prev_beam_indices = torch.div(best_indices, self.tgt_vocab.token2id.__len__(), rounding_mode='floor')
            token_indices = best_indices % self.tgt_vocab.token2id.__len__()

            new_ys = []
            new_scores = []
            active_beams = 0

            for i in range(beam_size):
                prev_idx = prev_beam_indices[i].item()
                token_idx = token_indices[i].item()
                score = best_scores[i].item()

                old_seq = ys[prev_idx]
                new_seq = torch.cat([old_seq, torch.tensor([token_idx], device=self.device)])

                if token_idx == self.eos_idx:
                    final_len = len(new_seq) - 1
                    normalized_score = score / (final_len ** alpha)
                    finished_sequences.append((normalized_score, new_seq))
                    new_scores.append(-float('inf'))
                    new_ys.append(new_seq)
                else:
                    new_scores.append(score)
                    new_ys.append(new_seq)
                    active_beams += 1

            beam_scores = torch.tensor(new_scores, device=self.device)
            ys = torch.stack(new_ys)

            if active_beams == 0:
                break

        # Select best sequence
        if len(finished_sequences) == 0:
            best_idx = beam_scores.argmax().item()
            final_seq = ys[best_idx]
        else:
            finished_sequences.sort(key=lambda x: x[0], reverse=True)
            final_seq = finished_sequences[0][1]

        final_ids = final_seq[1:].unsqueeze(0)

        # Re-run forward pass to get attention weights for the final sequence
        vis_input = final_seq[:-1].unsqueeze(0)
        if vis_input.size(1) == 0:
            vis_input = torch.tensor([[self.sos_idx]], device=self.device)

        vis_tgt_mask = self.model.make_tgt_mask(vis_input)
        original_src_mask = self.model.make_src_mask(src)
        original_memory = self.model.encoder(src, original_src_mask)

        _, final_attn = self.model.decoder(vis_input, original_memory, vis_tgt_mask, original_src_mask)

        # Format attention
        num_heads = self.config['model']['nhead']
        if final_attn is not None:
            tgt_len = vis_input.shape[1]
            src_len = src.shape[1]
            final_attn = final_attn.view(1, num_heads, tgt_len, src_len)
        else:
            final_attn = torch.zeros(1, num_heads, len(final_ids[0]), src.shape[1])

        return final_ids, final_attn

    def evaluate_loss(self, dataloader, criterion):
        """Calculates the validation loss.
        :param dataloader: Validation DataLoader.
        :param criterion: Loss function.
        :return: Average loss over the epoch.
        """
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src, trg in dataloader:
                src, trg = src.to(self.device), trg.to(self.device)

                if 'transformer' in self.model_name:
                    # Transformer uses teacher forcing via shifted inputs
                    trg_inp = trg[:, :-1]
                    trg_real = trg[:, 1:]
                    output, _ = self.model(src, trg_inp)
                    output = output.contiguous().view(-1, output.shape[-1])
                    trg_real = trg_real.contiguous().view(-1)
                else:
                    # RNN evaluation (teacher_forcing=0 means purely autoregressive usually, 
                    # but here likely using standard forward pass logic for loss calc)
                    output = self.model(src, trg, teacher_forcing_ratio=0)
                    output = output[:, 1:].contiguous().view(-1, output.shape[-1])
                    trg_real = trg[:, 1:].contiguous().view(-1)

                loss = criterion(output, trg_real)
                epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def generate_translations(self, dataloader, calculate_sentence_metrics=False, use_beam_search=True):
        """Runs inference to generate translations and compute metrics.
        :param dataloader: Test DataLoader.
        :param calculate_sentence_metrics: If True, returns per-sentence BLEU/ROUGE statistics.
        :param use_beam_search: If True, uses beam search; otherwise greedy decoding.
        :return: Tuple containing sources, hypotheses, references, attentions, stats, and scores.
        """
        self.model.eval()
        sources, hypotheses, references, all_attentions = [], [], [], []
        inference_times, sentence_bleus, sentence_rouges = [], [], []

        # Beam search is generally only viable for batch_size=1
        can_use_beam = use_beam_search and (dataloader.batch_size == 1)
        if use_beam_search and not can_use_beam:
            print("Warning: Beam search requested but not applicable (requires LSTM model and batch_size=1). Fallback to Greedy.")

        with torch.no_grad():
            for src, trg in tqdm(dataloader, desc="Evaluating"):
                src = src.to(self.device)
                start_time = time.time()

                # Choose decoding strategy
                if can_use_beam:
                    if 'transformer' in self.model_name:
                        pred_ids, attns = self.beam_decode_transformer(src, beam_size=5, alpha=0.7)
                    else:
                        pred_ids, attns = self.beam_decode_rnn(src, beam_size=5, alpha=0.7)
                else:
                    if 'transformer' in self.model_name:
                        pred_ids, attns = self.greedy_decode_transformer(src)
                    else:
                        pred_ids, attns = self.greedy_decode_rnn(src)

                inference_times.append(time.time() - start_time)

                # Decode IDs to text
                for i in range(src.shape[0]):
                    hyp_tokens = self.tgt_vocab.decode_ids(pred_ids[i].cpu().numpy().tolist())
                    ref_tokens = self.tgt_vocab.decode_ids(trg[i].cpu().numpy().tolist())
                    src_tokens = self.src_vocab.decode_ids(src[i].cpu().numpy().tolist())

                    hypotheses.append(hyp_tokens)
                    references.append(ref_tokens)
                    sources.append(src_tokens)
                    all_attentions.append(attns[i].cpu())

                    if calculate_sentence_metrics:
                        sentence_bleus.append(self.bleu_calc.compute_sentence_bleu(hyp_tokens, ref_tokens))
                        sentence_rouges.append(self.rouge_calc.compute_sentence_rouge(hyp_tokens, ref_tokens))

        # Calculate Corpus-level metrics
        corpus_bleu = self.bleu_calc.compute_bleu(hypotheses, references)
        corpus_rouge = self.rouge_calc.compute_corpus_rouge(hypotheses, references)
        time_stats = (np.mean(inference_times), np.std(inference_times))

        sentence_metrics = {}
        if calculate_sentence_metrics:
            sentence_metrics['bleu_mean_std'] = (np.mean(sentence_bleus), np.std(sentence_bleus))
            sentence_metrics['rouge_mean_std'] = (np.mean(sentence_rouges), np.std(sentence_rouges))

        return sources, hypotheses, references, all_attentions, time_stats, corpus_bleu, corpus_rouge, sentence_metrics
