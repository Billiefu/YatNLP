"""
Copyright (C) 2025 Fu Tszkok

:module: lstm
:function: Implements various LSTM-based Seq2Seq models (Naive/Standard, Bi-directional/Stacked) with Attention.
:author: Fu Tszkok
:date: 2025-12-23
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NaiveLSTM(nn.Module):
    """Manually implements a single LSTM cell (Long Short-Term Memory)."""

    def __init__(self, input_size, hidden_size):
        """Initializes the NaiveLSTM cell.
        :param input_size: The number of expected features in the input x.
        :param hidden_size: The number of features in the hidden state h.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define linear transformations for Input (i), Forget (f), Output (o) gates, and Cell (c) candidate
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)

        # Initialize forget gate biases to 1.0 to encourage remembering information early in training
        self.W_f.bias.data.fill_(1.0)
        self.U_f.bias.data.fill_(1.0)

    def forward(self, inputs, H_prev, C_prev):
        """Performs the forward pass for a single time step.
        :param inputs: The input tensor at current time step [batch_size, input_size].
        :param H_prev: The hidden state from the previous time step [batch_size, hidden_size].
        :param C_prev: The cell state from the previous time step [batch_size, hidden_size].
        :return: A tuple (H_new, C_new).
        """
        # Calculate gate activations using Sigmoid
        I = torch.sigmoid(self.W_i(inputs) + self.U_i(H_prev))
        F_gate = torch.sigmoid(self.W_f(inputs) + self.U_f(H_prev))
        O = torch.sigmoid(self.W_o(inputs) + self.U_o(H_prev))

        # Calculate candidate cell state using Tanh
        C_tilda = torch.tanh(self.W_c(inputs) + self.U_c(H_prev))

        # Update cell state: C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
        C_new = F_gate * C_prev + I * C_tilda
        # Update hidden state: h_t = o_t * tanh(C_t)
        H_new = O * torch.tanh(C_new)

        return H_new, C_new


class BiLSTM(nn.Module):
    """Implements a Bidirectional LSTM layer using the NaiveLSTM cell."""

    def __init__(self, input_size, hidden_size, dropout=0.0):
        """Initializes the BiLSTM layer.
        :param input_size: The number of expected features in the input.
        :param hidden_size: The number of features in the hidden state.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fwd_cell = NaiveLSTM(input_size, hidden_size)
        self.bwd_cell = NaiveLSTM(input_size, hidden_size)

    def forward(self, x):
        """Processes the input sequence in both forward and backward directions.
        :param x: Input sequence tensor [batch_size, seq_len, input_size].
        :return: A tuple (outputs, final_hidden, final_cell).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden and cell states for both directions
        H_fwd, C_fwd = torch.zeros(batch_size, self.hidden_size, device=device), torch.zeros(batch_size, self.hidden_size, device=device)
        H_bwd, C_bwd = torch.zeros(batch_size, self.hidden_size, device=device), torch.zeros(batch_size, self.hidden_size, device=device)

        outputs_fwd, outputs_bwd = [], []

        # Forward pass loop
        for t in range(seq_len):
            H_fwd, C_fwd = self.fwd_cell(x[:, t, :], H_fwd, C_fwd)
            outputs_fwd.append(H_fwd)

        # Backward pass loop
        for t in range(seq_len - 1, -1, -1):
            H_bwd, C_bwd = self.bwd_cell(x[:, t, :], H_bwd, C_bwd)
            outputs_bwd.insert(0, H_bwd)

        # Concatenate forward and backward outputs
        out_fwd, out_bwd = torch.stack(outputs_fwd, dim=1), torch.stack(outputs_bwd, dim=1)
        outputs = self.dropout(torch.cat([out_fwd, out_bwd], dim=2))

        # Concatenate final hidden and cell states
        final_hidden, final_cell = torch.cat([H_fwd, H_bwd], dim=1), torch.cat([C_fwd, C_bwd], dim=1)
        return outputs, final_hidden, final_cell


class Attention(nn.Module):
    """Implements Attention Mechanism (Dot, Multiplicative, Additive)."""

    def __init__(self, method, enc_out_dim, dec_hid_dim):
        """Initializes the Attention module.
        :param method: Alignment function ('dot', 'multiplicative', 'additive').
        :param enc_out_dim: Dimensionality of encoder output.
        :param dec_hid_dim: Dimensionality of decoder hidden state.
        """
        super().__init__()
        self.method = method
        self.dec_hid_dim = dec_hid_dim
        if method not in ['dot', 'multiplicative', 'additive']:
            raise ValueError(f"Unknown attention method: {self.method}")

        if method == 'multiplicative':
            self.W = nn.Linear(dec_hid_dim, enc_out_dim, bias=False)
        elif method == 'additive':
            self.W = nn.Linear(dec_hid_dim + enc_out_dim, dec_hid_dim, bias=False)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """Calculates attention weights.
        :param hidden: Current decoder hidden state [batch_size, dec_hid_dim].
        :param encoder_outputs: All encoder outputs [batch_size, src_len, enc_out_dim].
        :return: Attention weights [batch_size, src_len].
        """
        hidden = hidden.unsqueeze(1)  # [batch, 1, dec_hid]

        if self.method == 'dot':
            # Score = H_dec * H_enc^T / sqrt(dim)
            attn_energies = torch.bmm(hidden, encoder_outputs.permute(0, 2, 1)) / math.sqrt(self.dec_hid_dim)
        elif self.method == 'multiplicative':
            # Score = H_dec * W * H_enc^T
            attn_energies = torch.bmm(self.W(hidden), encoder_outputs.permute(0, 2, 1))
        elif self.method == 'additive':
            # Score = v^T * tanh(W * [H_dec; H_enc])
            src_len = encoder_outputs.shape[1]
            hidden_expanded = hidden.repeat(1, src_len, 1)
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = torch.tanh(self.W(combined))
            attn_energies = self.v(energy).permute(0, 2, 1)

        return F.softmax(attn_energies, dim=2)


class UnstackedBidirectionalEncoder(nn.Module):
    """Encoder using manually implemented BiLSTM (single layer)."""

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dropout):
        """Initializes the encoder.
        :param input_dim: Size of the source vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_hid_dim: Size of the hidden state.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = BiLSTM(emb_dim, enc_hid_dim, dropout)
        self.ln = nn.LayerNorm(enc_hid_dim * 2)

    def forward(self, src):
        """Encodes the source sequence.
        :param src: Source sequence tensor [batch_size, seq_len].
        :return: Tuple (outputs, hidden, cell).
        """
        embedded = self.dropout(self.embedding(src))
        outputs, hidden, cell = self.rnn(embedded)
        outputs = self.ln(outputs)
        return outputs, hidden, cell


class StandardUnstackedBidirectionalEncoder(nn.Module):
    """Encoder using PyTorch's native nn.LSTM (Bidirectional)."""

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dropout):
        """Initializes the standard encoder.
        :param input_dim: Size of the source vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_hid_dim: Size of the hidden state.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(enc_hid_dim * 2)

    def forward(self, src):
        """Encodes the source sequence using nn.LSTM.
        :param src: Source sequence tensor [batch_size, seq_len].
        :return: Tuple (outputs, hidden_cat, cell_cat).
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = self.ln(outputs)

        # Concatenate the hidden/cell states from forward and backward directions
        # hidden shape from nn.LSTM: [num_layers * num_directions, batch, hid_dim]
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell_cat = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)
        return outputs, hidden_cat, cell_cat


class StackedUnidirectionalEncoder(nn.Module):
    """Encoder using two stacked manually implemented NaiveLSTM layers."""

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dropout):
        """Initializes the stacked encoder.
        :param input_dim: Size of the source vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_hid_dim: Size of the hidden state.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer1 = NaiveLSTM(emb_dim, enc_hid_dim)
        self.layer2 = NaiveLSTM(enc_hid_dim, enc_hid_dim)

        self.ln1 = nn.LayerNorm(enc_hid_dim)
        self.ln2 = nn.LayerNorm(enc_hid_dim)

    def forward(self, src):
        """Encodes the source sequence through stacked layers.
        :param src: Source sequence tensor [batch_size, seq_len].
        :return: Tuple (final_outputs, final_hidden, final_cell).
        """
        batch_size, seq_len = src.shape
        device = src.device
        embedded = self.dropout(self.embedding(src))

        # Initialize states for both layers
        h1, c1 = torch.zeros(batch_size, self.layer1.hidden_size, device=device), torch.zeros(batch_size, self.layer1.hidden_size, device=device)
        h2, c2 = torch.zeros(batch_size, self.layer2.hidden_size, device=device), torch.zeros(batch_size, self.layer2.hidden_size, device=device)
        outputs = []

        # Process sequence step-by-step
        for t in range(seq_len):
            h1, c1 = self.layer1(embedded[:, t, :], h1, c1)
            # Pass layer1 output to layer2
            h2, c2 = self.layer2(self.dropout(self.ln1(h1)), h2, c2)
            outputs.append(self.ln2(h2))

        final_outputs = torch.stack(outputs, dim=1)
        final_hidden = torch.cat((h1, h2), dim=1)
        final_cell = torch.cat((c1, c2), dim=1)
        return final_outputs, final_hidden, final_cell


class StandardStackedUnidirectionalEncoder(nn.Module):
    """Encoder using PyTorch's native nn.LSTM (Stacked, 2 layers)."""

    def __init__(self, input_dim, emb_dim, enc_hid_dim, dropout):
        """Initializes the standard stacked encoder.
        :param input_dim: Size of the source vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_hid_dim: Size of the hidden state.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(enc_hid_dim)

    def forward(self, src):
        """Encodes the source sequence using stacked nn.LSTM.
        :param src: Source sequence tensor [batch_size, seq_len].
        :return: Tuple (outputs, hidden, cell).
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = self.ln(outputs)
        return outputs, hidden, cell


class UnstackedUnidirectionalDecoder(nn.Module):
    """Decoder for the bidirectional encoder setup (Single Layer NaiveLSTM)."""

    def __init__(self, output_dim, emb_dim, enc_out_dim, dec_hid_dim, dropout, attention):
        """Initializes the decoder.
        :param output_dim: Size of the target vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_out_dim: Dimension of encoder output (often 2*hidden for BiLSTM).
        :param dec_hid_dim: Dimension of decoder hidden state.
        :param dropout: Dropout probability.
        :param attention: The attention module instance.
        """
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn_cell = NaiveLSTM(emb_dim + enc_out_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_out_dim + dec_hid_dim + emb_dim, output_dim)
        self.ln = nn.LayerNorm(dec_hid_dim)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        """Performs a single decoding step.
        :param input_step: Input token index [batch_size].
        :param hidden: Previous hidden state.
        :param cell: Previous cell state.
        :param encoder_outputs: Outputs from the encoder.
        :return: Tuple (prediction, hidden, cell, attn_weights).
        """
        embedded = self.dropout(self.embedding(input_step.unsqueeze(1)))  # [batch, 1, emb]

        # Calculate attention weights and context vector
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)  # [batch, enc_out]

        # Concatenate embedding and context for RNN input
        rnn_input = torch.cat((embedded.squeeze(1), context), dim=1)
        hidden, cell = self.rnn_cell(rnn_input, hidden, cell)

        # Prepare input for final prediction layer
        hidden_norm = self.ln(hidden)
        prediction_input = torch.cat((embedded.squeeze(1), hidden_norm, context), dim=1)
        prediction = self.fc_out(prediction_input)

        return prediction, hidden, cell, attn_weights.squeeze(1)


class StandardUnstackedUnidirectionalDecoder(nn.Module):
    """Decoder for the bidirectional encoder setup (Single Layer nn.LSTMCell)."""

    def __init__(self, output_dim, emb_dim, enc_out_dim, dec_hid_dim, dropout, attention):
        """Initializes the standard decoder.
        :param output_dim: Size of the target vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_out_dim: Dimension of encoder output.
        :param dec_hid_dim: Dimension of decoder hidden state.
        :param dropout: Dropout probability.
        :param attention: The attention module instance.
        """
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn_cell = nn.LSTMCell(emb_dim + enc_out_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_out_dim + dec_hid_dim + emb_dim, output_dim)
        self.ln = nn.LayerNorm(dec_hid_dim)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        """Performs a single decoding step using nn.LSTMCell.
        :param input_step: Input token index [batch_size].
        :param hidden: Previous hidden state.
        :param cell: Previous cell state.
        :param encoder_outputs: Outputs from the encoder.
        :return: Tuple (prediction, hidden, cell, attn_weights).
        """
        embedded = self.dropout(self.embedding(input_step.unsqueeze(1)))

        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)

        rnn_input = torch.cat((embedded.squeeze(1), context), dim=1)
        hidden, cell = self.rnn_cell(rnn_input, (hidden, cell))

        hidden_norm = self.ln(hidden)
        prediction_input = torch.cat((embedded.squeeze(1), hidden_norm, context), dim=1)
        prediction = self.fc_out(prediction_input)

        return prediction, hidden, cell, attn_weights.squeeze(1)


class StackedUnidirectionalDecoder(nn.Module):
    """Decoder for the stacked encoder setup (2 Layers NaiveLSTM)."""

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        """Initializes the stacked decoder.
        :param output_dim: Size of the target vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_hid_dim: Dimension of encoder hidden state.
        :param dec_hid_dim: Dimension of decoder hidden state.
        :param dropout: Dropout probability.
        :param attention: The attention module instance.
        """
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer1 = NaiveLSTM(emb_dim + enc_hid_dim, dec_hid_dim)
        self.layer2 = NaiveLSTM(dec_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)

        self.ln1 = nn.LayerNorm(dec_hid_dim)
        self.ln2 = nn.LayerNorm(dec_hid_dim)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        """Performs a single decoding step through stacked layers.
        :param input_step: Input token index [batch_size].
        :param hidden: Concatenated previous hidden states of layers [batch_size, 2*dec_hid].
        :param cell: Concatenated previous cell states of layers [batch_size, 2*dec_hid].
        :param encoder_outputs: Outputs from the encoder.
        :return: Tuple (prediction, new_hidden, new_cell, attn_weights).
        """
        # Split hidden state for the two layers
        hid_dim = hidden.shape[1] // 2
        h1, h2 = hidden[:, :hid_dim], hidden[:, hid_dim:]
        c1, c2 = cell[:, :hid_dim], cell[:, hid_dim:]

        embedded = self.dropout(self.embedding(input_step.unsqueeze(1)))

        # Calculate attention using the top layer's hidden state
        attn_weights = self.attention(h2, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)

        # Layer 1
        rnn_input = torch.cat((embedded.squeeze(1), context.squeeze(1)), dim=1)
        h1, c1 = self.layer1(rnn_input, h1, c1)

        # Layer 2
        h2, c2 = self.layer2(self.dropout(self.ln1(h1)), h2, c2)

        # Final prediction
        prediction_input = torch.cat((embedded.squeeze(1), self.ln2(h2), context.squeeze(1)), dim=1)
        prediction = self.fc_out(prediction_input)

        new_hidden = torch.cat((h1, h2), dim=1)
        new_cell = torch.cat((c1, c2), dim=1)
        return prediction, new_hidden, new_cell, attn_weights.squeeze(1)


class StandardStackedUnidirectionalDecoder(nn.Module):
    """Decoder for the stacked encoder setup (2 Layers nn.LSTMCell)."""

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        """Initializes the standard stacked decoder.
        :param output_dim: Size of the target vocabulary.
        :param emb_dim: Size of the embedding vector.
        :param enc_hid_dim: Dimension of encoder hidden state.
        :param dec_hid_dim: Dimension of decoder hidden state.
        :param dropout: Dropout probability.
        :param attention: The attention module instance.
        """
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer1 = nn.LSTMCell(emb_dim + enc_hid_dim, dec_hid_dim)
        self.layer2 = nn.LSTMCell(dec_hid_dim, dec_hid_dim)

        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.ln1 = nn.LayerNorm(dec_hid_dim)
        self.ln2 = nn.LayerNorm(dec_hid_dim)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        """Performs a single decoding step using stacked nn.LSTMCell.
        :param input_step: Input token index [batch_size].
        :param hidden: Previous hidden states [2, batch_size, dec_hid].
        :param cell: Previous cell states [2, batch_size, dec_hid].
        :param encoder_outputs: Outputs from the encoder.
        :return: Tuple (prediction, new_hidden, new_cell, attn_weights).
        """
        h1, h2 = hidden[0], hidden[1]
        c1, c2 = cell[0], cell[1]

        embedded = self.dropout(self.embedding(input_step.unsqueeze(1)))

        attn_weights = self.attention(h2, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)

        # Layer 1
        rnn_input = torch.cat((embedded.squeeze(1), context), dim=1)
        h1_new, c1_new = self.layer1(rnn_input, (h1, c1))

        # Layer 2
        l1_out = self.dropout(self.ln1(h1_new))
        h2_new, c2_new = self.layer2(l1_out, (h2, c2))

        # Final prediction
        prediction_input = torch.cat((embedded.squeeze(1), self.ln2(h2_new), context), dim=1)
        prediction = self.fc_out(prediction_input)

        new_hidden = torch.stack([h1_new, h2_new], dim=0)
        new_cell = torch.stack([c1_new, c2_new], dim=0)

        return prediction, new_hidden, new_cell, attn_weights.squeeze(1)


class Seq2SeqRNN(nn.Module):
    """Encapsulates the Encoder-Decoder architecture for Sequence-to-Sequence learning."""

    def __init__(self, encoder, decoder, device):
        """Initializes the Seq2Seq model.
        :param encoder: The encoder module.
        :param decoder: The decoder module.
        :param device: The computing device (CPU/GPU).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """Runs the Seq2Seq model on a batch of data.
        :param src: Source sequence tensor.
        :param trg: Target sequence tensor.
        :param teacher_forcing_ratio: Probability of using teacher forcing.
        :return: Output tensor containing prediction logits [batch, trg_len, vocab_size].
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to the decoder is the <sos> token
        input_step = trg[:, 0]

        for t in range(1, trg_len):
            # Decode one step
            output, hidden, cell, _ = self.decoder(input_step, hidden, cell, encoder_outputs)

            # Store prediction
            outputs[:, t] = output

            # Decide whether to use teacher forcing or the model's own prediction
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input_step = trg[:, t] if teacher_force else top1

        return outputs


def build_model(config, src_vocab_size, tgt_vocab_size, device):
    """Constructs the Seq2Seq model based on the configuration dictionary.
    :param config: Dictionary containing hyperparameters (emb_size, hid_dim, etc.).
    :param src_vocab_size: Size of source vocabulary.
    :param tgt_vocab_size: Size of target vocabulary.
    :param device: Computing device.
    :return: An initialized Seq2SeqRNN model.
    """
    ENC_EMB_DIM, DEC_EMB_DIM = config.get('emb_size'), config.get('emb_size')
    HID_DIM = config.get('hid_dim', 512)
    DROPOUT = config.get('dropout', 0.5)
    ATTN_METHOD = config.get('attn_method', 'dot')
    RNN_TYPE = config.get('rnn_type', 'stacked-lstm')
    USE_NATIVE = config.get('use_native', False)

    print(f"Building Model: Type={RNN_TYPE}, Native={USE_NATIVE}, Attention={ATTN_METHOD}")

    # Build model based on type (Bi-LSTM vs Stacked-LSTM) and implementation (Native vs Hand-crafted)
    if RNN_TYPE == 'bi-lstm':
        ENC_HID_DIM, DEC_HID_DIM, ENC_OUTPUT_DIM = HID_DIM, HID_DIM * 2, HID_DIM * 2
        attn = Attention(ATTN_METHOD, ENC_OUTPUT_DIM, DEC_HID_DIM)

        if USE_NATIVE:
            print("Using Hand-crafted NaiveLSTM implemented Bi-LSTM.")
            enc = UnstackedBidirectionalEncoder(src_vocab_size, ENC_EMB_DIM, ENC_HID_DIM, DROPOUT)
            dec = UnstackedUnidirectionalDecoder(tgt_vocab_size, DEC_EMB_DIM, ENC_OUTPUT_DIM, DEC_HID_DIM, DROPOUT, attn)
        else:
            print("Using PyTorch Standard nn.LSTM implemented Bi-LSTM.")
            enc = StandardUnstackedBidirectionalEncoder(src_vocab_size, ENC_EMB_DIM, ENC_HID_DIM, DROPOUT)
            dec = StandardUnstackedUnidirectionalDecoder(tgt_vocab_size, DEC_EMB_DIM, ENC_OUTPUT_DIM, DEC_HID_DIM, DROPOUT, attn)

    elif RNN_TYPE == 'stacked-lstm':
        ENC_HID_DIM, DEC_HID_DIM, ENC_OUTPUT_DIM = HID_DIM, HID_DIM, HID_DIM
        attn = Attention(ATTN_METHOD, ENC_OUTPUT_DIM, DEC_HID_DIM)

        if USE_NATIVE:
            print("Using Hand-crafted NaiveLSTM implemented Stacked-LSTM.")
            enc = StackedUnidirectionalEncoder(src_vocab_size, ENC_EMB_DIM, ENC_HID_DIM, DROPOUT)
            dec = StackedUnidirectionalDecoder(tgt_vocab_size, DEC_EMB_DIM, ENC_OUTPUT_DIM, DEC_HID_DIM, DROPOUT, attn)
        else:
            print("Using PyTorch Standard nn.LSTM implemented Stacked-LSTM.")
            enc = StandardStackedUnidirectionalEncoder(src_vocab_size, ENC_EMB_DIM, ENC_HID_DIM, DROPOUT)
            dec = StandardStackedUnidirectionalDecoder(tgt_vocab_size, DEC_EMB_DIM, ENC_OUTPUT_DIM, DEC_HID_DIM, DROPOUT, attn)

    else:
        raise ValueError(f"Unknown RNN_TYPE {RNN_TYPE}")

    model = Seq2SeqRNN(enc, dec, device).to(device)

    # Initialize parameters
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)

    # Specific initialization for forget gate biases
    if USE_NATIVE:
        if hasattr(model.encoder, 'layer1'):
            model.encoder.layer1.W_f.bias.data.fill_(1.0)
            model.encoder.layer1.U_f.bias.data.fill_(1.0)
            model.encoder.layer2.W_f.bias.data.fill_(1.0)
            model.encoder.layer2.U_f.bias.data.fill_(1.0)
        if hasattr(model.decoder, 'layer1'):
            model.decoder.layer1.W_f.bias.data.fill_(1.0)
            model.decoder.layer1.U_f.bias.data.fill_(1.0)
            model.decoder.layer2.W_f.bias.data.fill_(1.0)
            model.decoder.layer2.U_f.bias.data.fill_(1.0)
    else:
        for name, param in model.named_parameters():
            if 'lstm' in name and 'bias' in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

    return model
