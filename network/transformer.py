"""
Copyright (C) 2025 Fu Tszkok

:module: transformer
:function: Implements the Transformer architecture, including Standard and Optimized (GQA, Sparse) variants.
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
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""

    def __init__(self, hidden_size, max_len=5000):
        """Initializes the sinusoidal positional encoding.
        :param hidden_size: The embedding dimension.
        :param max_len: The maximum length of the incoming sequence.
        """
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

        # Apply Sine to even indices and Cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, X):
        """Adds positional encoding to the input embeddings.
        :param X: Input tensor [batch_size, seq_len, hidden_size].
        :return: Output tensor with positional information added.
        """
        return X + self.pe[:, :X.size(1), :]


class PositionWiseFNN(nn.Module):
    """Implements the Position-wise Feed-Forward Network."""

    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        """Initializes the FFN.
        :param hidden_size: Input and output dimensionality.
        :param intermediate_size: Hidden layer dimensionality (usually 4x hidden_size).
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """Passes the input through the FFN.
        :param X: Input tensor.
        :return: Output tensor.
        """
        return self.dense2(self.dropout(self.relu(self.dense1(X))))


class AddNorm(nn.Module):
    """Implements the Residual Connection followed by Layer Normalization."""

    def __init__(self, normalized_shape, dropout=0.1):
        """Initializes the AddNorm layer.
        :param normalized_shape: Input shape for LayerNorm.
        :param dropout: Dropout probability applied to the sub-layer output.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """Applies Add & Norm: LayerNorm(X + Dropout(Y)).
        :param X: The input to the residual connection (identity).
        :param Y: The output from the sub-layer (Attention or FFN).
        :return: Normalized tensor.
        """
        return self.layer_norm(self.dropout(Y) + X)


class MultiHeadAttention(nn.Module):
    """Implements standard Multi-Head Attention mechanism."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        """Initializes MHA.
        :param hidden_size: The embedding dimension.
        :param num_heads: The number of attention heads.
        :param dropout: Dropout probability.
        """
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear projections for Query, Key, Value, and Output
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_qkv(self, states):
        """Reshapes and permutes states for multi-head processing.
        :param states: Tensor [batch_size, seq_len, hidden_size].
        :return: Tensor [batch_size, num_heads, seq_len, head_dim].
        """
        states = states.reshape(states.shape[0], states.shape[1], self.num_heads, -1)
        states = states.permute(0, 2, 1, 3)
        return states.reshape(-1, states.shape[2], states.shape[3])

    def transpose_output(self, states):
        """Restores the original shape from multi-head output.
        :param states: Tensor [batch_size * num_heads, seq_len, head_dim].
        :return: Tensor [batch_size, seq_len, hidden_size].
        """
        states = states.reshape(-1, self.num_heads, states.shape[1], states.shape[2])
        states = states.permute(0, 2, 1, 3)
        return states.reshape(states.shape[0], states.shape[1], -1)

    def forward(self, query, key, value, mask=None):
        """Computes Scaled Dot-Product Attention.
        :param query: Query tensor.
        :param key: Key tensor.
        :param value: Value tensor.
        :param mask: Optional mask tensor (0 for masked positions).
        :return: Tuple (output, attn_weights).
        """
        # Linear projection and reshape for multi-head
        q = self.transpose_qkv(self.W_q(query))
        k = self.transpose_qkv(self.W_k(key))
        v = self.transpose_qkv(self.W_v(value))

        # Calculate attention scores: Q * K^T / sqrt(d_k)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.repeat(1, self.num_heads, 1, 1)
                mask = mask.reshape(-1, mask.shape[2], mask.shape[3])
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.bmm(self.dropout(attn_weights), v)

        # Recombine heads and apply output projection
        output = self.transpose_output(attn_output)
        return self.W_o(output), attn_weights


class EncoderLayer(nn.Module):
    """A single layer of the Transformer Encoder."""

    def __init__(self, hidden_size, num_heads, ffn_dim, dropout):
        """Initializes the Encoder Layer.
        :param hidden_size: Hidden dimension.
        :param num_heads: Number of attention heads.
        :param ffn_dim: Intermediate FFN dimension.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.add_norm1 = AddNorm(hidden_size, dropout)
        self.ffn = PositionWiseFNN(hidden_size, ffn_dim, dropout)
        self.add_norm2 = AddNorm(hidden_size, dropout)

    def forward(self, X, src_mask):
        """Processes input through Self-Attention and FFN.
        :param X: Input tensor.
        :param src_mask: Mask for source sequence.
        :return: Transformed tensor.
        """
        # Sub-layer 1: Self-Attention
        attn_output, _ = self.self_attn(X, X, X, src_mask)
        X = self.add_norm1(X, attn_output)

        # Sub-layer 2: FFN
        X = self.add_norm2(X, self.ffn(X))
        return X


class DecoderLayer(nn.Module):
    """A single layer of the Transformer Decoder."""

    def __init__(self, hidden_size, num_heads, ffn_dim, dropout):
        """Initializes the Decoder Layer.
        :param hidden_size: Hidden dimension.
        :param num_heads: Number of attention heads.
        :param ffn_dim: Intermediate FFN dimension.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.add_norm1 = AddNorm(hidden_size, dropout)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.add_norm2 = AddNorm(hidden_size, dropout)
        self.ffn = PositionWiseFNN(hidden_size, ffn_dim, dropout)
        self.add_norm3 = AddNorm(hidden_size, dropout)

    def forward(self, X, memory, tgt_mask, src_mask):
        """Processes input through Self-Attention, Cross-Attention, and FFN.
        :param X: Target input tensor.
        :param memory: Encoder output tensor.
        :param tgt_mask: Mask for target sequence (look-ahead mask).
        :param src_mask: Mask for source sequence.
        :return: Tuple (Output, Cross-Attention Weights).
        """
        # Sub-layer 1: Masked Self-Attention
        self_attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, self_attn_output)

        # Sub-layer 2: Cross-Attention (Query from Decoder, Key/Value from Encoder)
        cross_attn_output, cross_attn_weights = self.cross_attn(X, memory, memory, src_mask)
        X = self.add_norm2(X, cross_attn_output)

        # Sub-layer 3: FFN
        X = self.add_norm3(X, self.ffn(X))
        return X, cross_attn_weights


class TransformerEncoder(nn.Module):
    """The standard Transformer Encoder consisting of stacked layers."""

    def __init__(self, input_dim, hidden_size, num_layers, num_heads, ffn_dim, dropout, max_len=5000):
        """Initializes the Transformer Encoder."""
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.pe = PositionalEncoding(hidden_size, max_len)
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, src, src_mask):
        """Encodes the source sequence."""
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """The standard Transformer Decoder consisting of stacked layers."""

    def __init__(self, output_dim, hidden_size, num_layers, num_heads, ffn_dim, dropout, max_len=5000):
        """Initializes the Transformer Decoder."""
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_size)
        self.pe = PositionalEncoding(hidden_size, max_len)
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        """Decodes the target sequence using the encoder memory."""
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pe(x)
        x = self.dropout(x)
        cross_attn_weights = None
        for layer in self.layers:
            x, cross_attn_weights = layer(x, memory, tgt_mask, src_mask)
        x = self.norm(x)
        output = self.fc_out(x)
        return output, cross_attn_weights


class OptimalMultiHeadAttention(nn.Module):
    """Implements optimized Attention: Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Sparse Attention."""

    def __init__(self, hidden_size, num_heads, num_kv_heads=None, dropout=0.1, use_sparse=False, window_size=5):
        """Initializes Optimal Attention.
        :param hidden_size: Embedding dimension.
        :param num_heads: Number of query heads.
        :param num_kv_heads: Number of Key/Value heads (GQA/MQA). If None, defaults to num_heads (Standard MHA).
        :param use_sparse: Boolean to enable sparse (local window) attention.
        :param window_size: The window size for sparse attention.
        """
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Determine KV heads configuration
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.use_sparse = use_sparse
        self.window_size = window_size

        # Projections (Note: K/V output dim depends on num_kv_heads)
        self.W_q = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def generate_sparse_mask(self, seq_len, device):
        """Creates a local window mask for sparse attention.
        :param seq_len: Length of the sequence.
        :return: Boolean mask where True indicates allowed positions.
        """
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=-self.window_size) & torch.tril(mask, diagonal=self.window_size)
        return mask

    def forward(self, query, key, value, mask=None):
        """Computes attention with GQA/MQA and Sparse logic.
        :param query: Query tensor.
        :param key: Key tensor.
        :param value: Value tensor.
        :param mask: Standard attention mask.
        :return: Tuple (output, attn_weights).
        """
        batch_size, seq_len_q, _ = query.shape
        batch_size, seq_len_k, _ = key.shape

        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # Reshape for multi-head: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Handle GQA/MQA: Expand K and V to match the number of Query heads
        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_groups, seq_len_k, self.head_dim)
            v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, self.num_kv_groups, seq_len_k, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len_k, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len_k, self.head_dim)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply provided mask (e.g., padding mask)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply Sparse Attention Mask if enabled
        if self.use_sparse and seq_len_q == seq_len_k and seq_len_q > self.window_size * 2:
            sparse_mask = self.generate_sparse_mask(seq_len_q, query.device)
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~sparse_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attn_weights), v)

        # Reshape back to [batch, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)
        return self.W_o(output), attn_weights


class OptimalEncoderLayer(nn.Module):
    """Encoder Layer utilizing the Optimal Attention mechanism."""

    def __init__(self, hidden_size, num_heads, ffn_dim, dropout, num_kv_heads=None, use_sparse=False):
        """Initializes the Optimal Encoder Layer."""
        super().__init__()
        self.self_attn = OptimalMultiHeadAttention(hidden_size, num_heads, num_kv_heads=num_kv_heads, dropout=dropout, use_sparse=use_sparse)
        self.add_norm1 = AddNorm(hidden_size, dropout)
        self.ffn = PositionWiseFNN(hidden_size, ffn_dim, dropout)
        self.add_norm2 = AddNorm(hidden_size, dropout)

    def forward(self, X, src_mask):
        """Processes input using Optimal Attention."""
        attn_output, _ = self.self_attn(X, X, X, src_mask)
        X = self.add_norm1(X, attn_output)
        X = self.add_norm2(X, self.ffn(X))
        return X


class OptimalDecoderLayer(nn.Module):
    """Decoder Layer utilizing the Optimal Attention mechanism."""

    def __init__(self, hidden_size, num_heads, ffn_dim, dropout, num_kv_heads=None):
        """Initializes the Optimal Decoder Layer.
        :note: Sparse attention is generally disabled in cross-attention for alignment accuracy.
        """
        super().__init__()
        # Self-attention in decoder usually requires full context or causal mask, sparse can be applied if careful
        self.self_attn = OptimalMultiHeadAttention(hidden_size, num_heads, num_kv_heads=num_kv_heads, dropout=dropout, use_sparse=False)
        self.add_norm1 = AddNorm(hidden_size, dropout)

        self.cross_attn = OptimalMultiHeadAttention(hidden_size, num_heads, num_kv_heads=num_kv_heads, dropout=dropout, use_sparse=False)
        self.add_norm2 = AddNorm(hidden_size, dropout)

        self.ffn = PositionWiseFNN(hidden_size, ffn_dim, dropout)
        self.add_norm3 = AddNorm(hidden_size, dropout)

    def forward(self, X, memory, tgt_mask, src_mask):
        """Processes input using Optimal Attention."""
        self_attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, self_attn_output)

        cross_attn_output, cross_attn_weights = self.cross_attn(X, memory, memory, src_mask)
        X = self.add_norm2(X, cross_attn_output)

        X = self.add_norm3(X, self.ffn(X))
        return X, cross_attn_weights


class OptimalTransformerEncoder(nn.Module):
    """Transformer Encoder built with Optimal Layers."""

    def __init__(self, input_dim, hidden_size, num_layers, num_heads, ffn_dim, dropout, num_kv_heads=None, use_sparse=False, max_len=5000):
        """Initializes the Optimal Encoder."""
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_size)
        self.pe = PositionalEncoding(hidden_size, max_len)
        self.layers = nn.ModuleList([
            OptimalEncoderLayer(hidden_size, num_heads, ffn_dim, dropout, num_kv_heads, use_sparse)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, src, src_mask):
        """Encodes the source sequence."""
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class OptimalTransformerDecoder(nn.Module):
    """Transformer Decoder built with Optimal Layers."""

    def __init__(self, output_dim, hidden_size, num_layers, num_heads, ffn_dim, dropout, num_kv_heads=None, max_len=5000):
        """Initializes the Optimal Decoder."""
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_size)
        self.pe = PositionalEncoding(hidden_size, max_len)
        self.layers = nn.ModuleList([
            OptimalDecoderLayer(hidden_size, num_heads, ffn_dim, dropout, num_kv_heads)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        """Decodes the target sequence."""
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pe(x)
        x = self.dropout(x)
        cross_attn_weights = None
        for layer in self.layers:
            x, cross_attn_weights = layer(x, memory, tgt_mask, src_mask)
        x = self.norm(x)
        output = self.fc_out(x)
        return output, cross_attn_weights


class Seq2SeqTransformer(nn.Module):
    """Encapsulates the complete Transformer Seq2Seq model."""

    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        """Initializes the Seq2Seq Transformer.
        :param encoder: Encoder module.
        :param decoder: Decoder module.
        :param src_pad_idx: Padding index for source sequences.
        :param tgt_pad_idx: Padding index for target sequences.
        :param device: Computing device.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        self._init_weights()

    def _init_weights(self):
        """Initializes parameters using Xavier Uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """Creates a mask for the source sequence to ignore padding tokens.
        :param src: Source tensor [batch_size, src_len].
        :return: Mask tensor [batch_size, 1, 1, src_len].
        """
        mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt):
        """Creates a combined mask for target sequence (Padding + Look-ahead).
        :param tgt: Target tensor [batch_size, tgt_len].
        :return: Boolean mask.
        """
        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.shape[1]
        tril_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()
        return pad_mask & tril_mask

    def forward(self, src, trg):
        """Runs the full Seq2Seq forward pass.
        :param src: Source batch.
        :param trg: Target batch.
        :return: Tuple (Prediction, Attention Weights).
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attn = self.decoder(trg, enc_src, tgt_mask, src_mask)
        return output, attn


def build_model(config, src_vocab_size, tgt_vocab_size, device):
    """Constructs the Transformer model based on configuration.
    :param config: Configuration dictionary.
    :param src_vocab_size: Source vocabulary size.
    :param tgt_vocab_size: Target vocabulary size.
    :param device: Computing device.
    :return: Initialized Seq2SeqTransformer model.
    """
    HID_DIM = config.get('emb_size', 512)
    ENC_LAYERS = config.get('enc_layers', 3)
    DEC_LAYERS = config.get('dec_layers', 3)
    HEADS = config.get('nhead', 8)
    FFN_DIM = config.get('ffn_dim', 2048)
    DROPOUT = config.get('dropout', 0.1)
    SRC_PAD_IDX = config.get('src_pad_idx', 0)
    TGT_PAD_IDX = config.get('tgt_pad_idx', 0)

    TRANSFORMER_TYPE = config.get('transformer_type', 'standard')

    if TRANSFORMER_TYPE == 'standard':
        print("[Info] Building STANDARD Transformer...")
        enc = TransformerEncoder(src_vocab_size, HID_DIM, ENC_LAYERS, HEADS, FFN_DIM, DROPOUT)
        dec = TransformerDecoder(tgt_vocab_size, HID_DIM, DEC_LAYERS, HEADS, FFN_DIM, DROPOUT)
    elif TRANSFORMER_TYPE == 'optimal':
        print("[Info] Building OPTIMAL Transformer (GQA/MQA/Sparse)...")
        NUM_KV_HEADS = config.get('num_kv_heads', 1)
        USE_SPARSE = config.get('use_sparse', True)
        print(f"       > GQA Config: Heads={HEADS}, KV_Heads={NUM_KV_HEADS}")
        print(f"       > Sparse Attention: {USE_SPARSE}")
        enc = OptimalTransformerEncoder(src_vocab_size, HID_DIM, ENC_LAYERS, HEADS, FFN_DIM, DROPOUT, num_kv_heads=NUM_KV_HEADS, use_sparse=USE_SPARSE)
        dec = OptimalTransformerDecoder(tgt_vocab_size, HID_DIM, DEC_LAYERS, HEADS, FFN_DIM, DROPOUT, num_kv_heads=NUM_KV_HEADS)
    else:
        raise ValueError(f"Unknown TRANSFORMER_TYPE: {TRANSFORMER_TYPE}")

    model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TGT_PAD_IDX, device).to(device)

    def init_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(init_weights)
    return model
