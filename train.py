"""
Copyright (C) 2025 Fu Tszkok

:module: train
:function: Handles the training loop, optimization, and learning rate scheduling for the Seq2Seq models.
:author: Fu Tszkok
:date: 2025-12-25
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class TrainingEngine:
    """Manages the training process for one epoch."""

    def __init__(self, model, device, config, vocab_size_tgt, pad_idx):
        """Initializes the Training Engine.
        :param model: The Seq2Seq model.
        :param device: Computing device.
        :param config: Configuration dictionary.
        :param vocab_size_tgt: Size of the target vocabulary.
        :param pad_idx: Padding index to ignore in loss calculation.
        """
        self.model = model
        self.device = device
        self.config = config
        self.pad_idx = pad_idx
        self.vocab_size_tgt = vocab_size_tgt
        self.model_name = config['model']['name']

        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_criterion(self):
        """Creates the loss function with optional label smoothing.
        :return: nn.CrossEntropyLoss instance.
        """
        label_smoothing = self.config['train'].get('label_smoothing', 0.0)
        return nn.CrossEntropyLoss(ignore_index=self.pad_idx, label_smoothing=label_smoothing)

    def _get_optimizer(self):
        """Configures the optimizer (Adam or AdamW).
        :return: Optimizer instance.
        """
        optimizer_name = self.config['train'].get('optimizer', 'adam').lower()
        lr = self.config['train']['lr']
        weight_decay = float(self.config['train'].get('weight_decay', 0))

        if optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            if 'transformer' in self.model_name:
                # Specific betas for Transformer stability
                return optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9,
                                  weight_decay=weight_decay)
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_scheduler(self):
        """Configures the Learning Rate Scheduler.
        :return: Scheduler instance (LambdaLR for Transformer, ReduceLROnPlateau for RNN).
        """
        if 'transformer' in self.model_name:
            warmup_epochs = 5

            # Custom warm-up schedule for Transformer
            def lr_lambda(epoch):
                current_epoch = epoch + 1
                if current_epoch <= warmup_epochs:
                    return current_epoch / warmup_epochs
                else:
                    return (warmup_epochs / current_epoch) ** 0.5

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    def _get_teacher_forcing_ratio(self, epoch_idx):
        """Calculates the teacher forcing ratio based on linear decay.
        :param epoch_idx: Current epoch index.
        :return: Float ratio [0.0, 1.0].
        """
        start = self.config['train'].get('tf_ratio_start', 1.0)
        end = self.config['train'].get('tf_ratio_end', 0.0)
        decay_epochs = self.config['train'].get('tf_decay_epochs', 10)

        if epoch_idx > decay_epochs:
            return end

        ratio = start - (start - end) * ((epoch_idx - 1) / decay_epochs)
        return max(end, ratio)

    def train_one_epoch(self, dataloader, epoch_idx):
        """Executes one training epoch.
        :param dataloader: Training DataLoader.
        :param epoch_idx: Current epoch index.
        :return: Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0

        # Update Teacher Forcing ratio
        tf_ratio = self._get_teacher_forcing_ratio(epoch_idx)

        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_idx} | LR: {current_lr:.6f} | Teacher Forcing Ratio: {tf_ratio:.2f}")

        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_idx}", unit="batch")

        for src, trg in pbar:
            src, trg = src.to(self.device), trg.to(self.device)

            self.optimizer.zero_grad()

            if 'transformer' in self.model_name:
                # Transformer Training: Shift target for teacher forcing input
                trg_inp = trg[:, :-1]
                output, _ = self.model(src, trg_inp)
            else:
                # RNN Training: Uses teacher forcing ratio inside the forward pass
                output = self.model(src, trg, teacher_forcing_ratio=tf_ratio)

            # Ignore the first token (SOS) in the output for loss calculation
            if 'transformer' not in self.model_name:
                output = output[:, 1:]

            # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
            output_reshaped = output.contiguous().view(-1, self.vocab_size_tgt)
            trg_real = trg[:, 1:].contiguous().view(-1)

            loss = self.criterion(output_reshaped, trg_real)

            loss.backward()

            # Gradient clipping
            clip_val = self.config['train'].get('clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

            self.optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return epoch_loss / len(dataloader)

    def step_scheduler(self, val_loss=None):
        """Steps the learning rate scheduler.
        :param val_loss: Validation loss (required for ReduceLROnPlateau).
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics=val_loss)
        else:
            self.scheduler.step()
