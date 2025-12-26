"""
Copyright (C) 2025 Fu Tszkok

:module: dataloader
:function: Handles data loading, preprocessing, tokenization, vocabulary construction, and batch creation for NMT.
:author: Fu Tszkok
:date: 2025-12-25
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import collections
import json
import os
import warnings
from functools import partial
from multiprocessing import Pool

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer import get_tokenizer

warnings.filterwarnings("ignore", category=UserWarning)

# Define special tokens used for padding and sequence boundaries
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# Define corresponding indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize_worker(item, src_tokenizer_type, tgt_tokenizer_type, src_model_path=None, tgt_model_path=None):
    """Worker function for multiprocessing tokenization.
    :param item: Dictionary containing 'zh' and 'en' raw strings.
    :param src_tokenizer_type: Type string for source tokenizer.
    :param tgt_tokenizer_type: Type string for target tokenizer.
    :param src_model_path: Path to load trained source tokenizer model (if applicable).
    :param tgt_model_path: Path to load trained target tokenizer model (if applicable).
    :return: The item dictionary updated with 'zh_tokens' and 'en_tokens'.
    """
    # Initialize source tokenizer
    if 'jieba' in src_tokenizer_type or 'nltk' in src_tokenizer_type or 'simple' in src_tokenizer_type:
        src_tok = get_tokenizer(src_tokenizer_type)
    else:
        src_tok = get_tokenizer(src_tokenizer_type)
        src_tok.load(src_model_path)

    # Initialize target tokenizer
    if 'jieba' in tgt_tokenizer_type or 'nltk' in tgt_tokenizer_type or 'simple' in tgt_tokenizer_type:
        tgt_tok = get_tokenizer(tgt_tokenizer_type)
    else:
        tgt_tok = get_tokenizer(tgt_tokenizer_type)
        tgt_tok.load(tgt_model_path)

    # Perform tokenization
    item['zh_tokens'] = src_tok.encode(item['zh'])
    item['en_tokens'] = tgt_tok.encode(item['en'])
    return item


class Vocabulary:
    """Manages the mapping between tokens and integer IDs."""

    def __init__(self, tokens=None, min_freq=0, specials=None):
        """Initializes the Vocabulary.
        :param tokens: List of all tokens in the dataset to build frequency map.
        :param min_freq: Minimum frequency for a token to be included in the vocab.
        :param specials: List of special tokens to ensure are at the beginning.
        """
        self.token2id = {}
        self.id2token = {}
        self.specials = specials if specials else [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # Register special tokens first
        for s in self.specials:
            self.add_token(s)

        # Register common tokens if they meet frequency threshold
        if tokens:
            counter = collections.Counter(tokens)
            sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])
            for token, freq in sorted_tokens:
                if freq >= min_freq:
                    self.add_token(token)

    def add_token(self, token):
        """Adds a token to the vocabulary if it doesn't exist.
        :param token: The string token to add.
        """
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.token2id)

    def __getitem__(self, token):
        """Retrieves the ID for a given token, returning UNK_IDX if not found.
        :param token: The token string.
        :return: Integer ID.
        """
        return self.token2id.get(token, self.token2id.get(UNK_TOKEN))

    def lookup_tokens(self, indices):
        """Converts a list of IDs back to tokens.
        :param indices: List of integer IDs.
        :return: List of token strings.
        """
        return [self.id2token.get(idx, UNK_TOKEN) for idx in indices]

    def decode_ids(self, ids, stop_at_eos=True, sep=" "):
        """Decodes a sequence of IDs into a readable string.
        :param ids: List or Tensor of token IDs.
        :param stop_at_eos: If True, stops decoding when EOS token is encountered.
        :param sep: Separator used to join tokens.
        :return: Decoded string.
        """
        tokens = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if stop_at_eos and i == EOS_IDX:
                break
            if i in [PAD_IDX, SOS_IDX, EOS_IDX]:
                continue
            tokens.append(self.id2token.get(i, UNK_TOKEN))
        return sep.join(tokens)


class NMTDataset(Dataset):
    """PyTorch Dataset for Neural Machine Translation tasks."""

    def __init__(self, data_path, config, src_vocab=None, tgt_vocab=None, is_train=True):
        """Initializes the dataset.
        :param data_path: Path to the .jsonl data file.
        :param config: Configuration dictionary.
        :param src_vocab: Pre-built source Vocabulary (required for validation/test).
        :param tgt_vocab: Pre-built target Vocabulary (required for validation/test).
        :param is_train: Boolean flag indicating if this is the training set.
        """
        self.data = self._read_file(data_path)
        self.is_train = is_train
        self.min_freq = config.get('min_freq', 1)
        self.config = config

        self.src_tokenizer_type = config.get('src_tokenizer', 'jieba')
        self.tgt_tokenizer_type = config.get('tgt_tokenizer', 'simple')

        # Paths for saving/loading trained tokenizer models (BPE/WordPiece)
        self.src_model_path = os.path.join(config.get('save_dir', '.'), f'src_{self.src_tokenizer_type}.json')
        self.tgt_model_path = os.path.join(config.get('save_dir', '.'), f'tgt_{self.tgt_tokenizer_type}.json')

        if self.is_train:
            self._prepare_tokenizers()

        print(f"Tokenizing {data_path} using {self.src_tokenizer_type} -> {self.tgt_tokenizer_type}...")
        self._tokenize()

        if self.is_train:
            assert src_vocab is None and tgt_vocab is None, "训练集应该从头构建词表"
            print(f"Building Vocabularies with min_freq={self.min_freq}...")
            self.src_vocab = self._build_vocab(lang='zh', min_freq=self.min_freq)
            self.tgt_vocab = self._build_vocab(lang='en', min_freq=self.min_freq)
        else:
            assert src_vocab is not None and tgt_vocab is not None, "测试/验证集必须传入训练好的词表"
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self._convert_tokens_to_ids()

        print(f"Dataset loaded. Size: {len(self.data)}")
        if self.is_train:
            print(f"Src Vocab: {len(self.src_vocab)}, Tgt Vocab: {len(self.tgt_vocab)}")

    def _read_file(self, file_path):
        """Reads JSONL file and normalizes keys.
        :param file_path: Path to file.
        :return: List of dictionaries with 'zh' and 'en' keys.
        """
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    # Support different key naming conventions
                    if 'zh' in obj and 'en' in obj:
                        data_list.append({'zh': obj['zh'], 'en': obj['en']})
                    elif 'src' in obj and 'tgt' in obj:
                        data_list.append({'zh': obj['src'], 'en': obj['tgt']})
                except json.JSONDecodeError:
                    continue
        return data_list

    def _prepare_tokenizers(self):
        """Trains statistical tokenizers (BPE/WordPiece) if needed."""
        vocab_size = self.config.get('src_vocab_size', 10000)

        # Train Source Tokenizer if it's learnable
        if self.src_tokenizer_type in ['bpe', 'wordpiece']:
            src_tok = get_tokenizer(self.src_tokenizer_type, vocab_size=vocab_size)
            if not os.path.exists(self.src_model_path):
                print(f"Training Source Tokenizer ({self.src_tokenizer_type}) with vocab_size={vocab_size}...")
                all_src_texts = [d['zh'] for d in self.data]
                src_tok.train(all_src_texts)
                src_tok.save(self.src_model_path)
            else:
                print(f"Loading Source Tokenizer from {self.src_model_path}")

        # Train Target Tokenizer if it's learnable
        vocab_size_tgt = self.config.get('tgt_vocab_size', 10000)
        if self.tgt_tokenizer_type in ['bpe', 'wordpiece']:
            tgt_tok = get_tokenizer(self.tgt_tokenizer_type, vocab_size=vocab_size_tgt)
            if not os.path.exists(self.tgt_model_path):
                print(f"Training Target Tokenizer ({self.tgt_tokenizer_type}) with vocab_size={vocab_size_tgt}...")
                all_tgt_texts = [d['en'] for d in self.data]
                tgt_tok.train(all_tgt_texts)
                tgt_tok.save(self.tgt_model_path)
            else:
                print(f"Loading Target Tokenizer from {self.tgt_model_path}")

    def _tokenize(self):
        """Tokenizes the dataset, using multiprocessing for large datasets."""
        worker_func = partial(tokenize_worker, src_tokenizer_type=self.src_tokenizer_type,
                              tgt_tokenizer_type=self.tgt_tokenizer_type, src_model_path=self.src_model_path,
                              tgt_model_path=self.tgt_model_path)

        # Use Multiprocessing for large datasets to speed up Jieba/NLTK
        if len(self.data) > 5000 and self.src_tokenizer_type not in ['bpe', 'wordpiece']:
            with Pool() as pool:
                tokenized_data = list(
                    tqdm(pool.imap(worker_func, self.data, chunksize=1000), total=len(self.data), desc="Tokenizing"))
            self.data = tokenized_data
        else:
            # Sequential processing (better for BPE/WordPiece which are already parallelized or fast)
            print("Using sequential processing for tokenization...")
            src_tok = get_tokenizer(self.src_tokenizer_type)
            if self.src_tokenizer_type in ['bpe', 'wordpiece']: src_tok.load(self.src_model_path)

            tgt_tok = get_tokenizer(self.tgt_tokenizer_type)
            if self.tgt_tokenizer_type in ['bpe', 'wordpiece']: tgt_tok.load(self.tgt_model_path)

            for item in tqdm(self.data, desc="Tokenizing"):
                item['zh_tokens'] = src_tok.encode(item['zh'])
                item['en_tokens'] = tgt_tok.encode(item['en'])

    def _build_vocab(self, lang='zh', min_freq=1):
        """Builds a Vocabulary object from tokenized data.
        :param lang: Language key ('zh' or 'en').
        :param min_freq: Minimum frequency threshold.
        :return: Vocabulary object.
        """
        all_tokens = []
        key = f'{lang}_tokens'
        for item in self.data:
            all_tokens.extend(item[key])
        return Vocabulary(all_tokens, min_freq=min_freq)

    def _convert_tokens_to_ids(self):
        """Converts token lists to integer ID lists, adding SOS and EOS tokens."""
        for item in self.data:
            src_tokens = item['zh_tokens']
            tgt_tokens = item['en_tokens']
            item['src_ids'] = [SOS_IDX] + [self.src_vocab[t] for t in src_tokens] + [EOS_IDX]
            item['tgt_ids'] = [SOS_IDX] + [self.tgt_vocab[t] for t in tgt_tokens] + [EOS_IDX]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['src_ids'], self.data[idx]['tgt_ids']


def collate_fn(batch):
    """Custom collate function to pad sequences in a batch.
    :param batch: List of tuples (src_ids, tgt_ids).
    :return: Tuple of padded tensors (src_batch_pad, tgt_batch_pad).
    """
    src_batch, tgt_batch = [], []
    for src_ids, tgt_ids in batch:
        src_batch.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))

    # Pad sequences to the length of the longest sequence in the batch using PAD_IDX
    src_batch_pad = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch_pad, tgt_batch_pad


def get_dataloaders(config):
    """Creates DataLoaders for train, validation, and test sets.
    :param config: Configuration dictionary.
    :return: Tuple (train_loader, val_loader, test_loader, src_vocab, tgt_vocab).
    """
    data_cfg = config

    # Initialize Train Dataset (builds vocab)
    train_dataset = NMTDataset(config['train_path'], data_cfg, is_train=True)

    # Initialize Val/Test Datasets (reuse vocab from Train)
    val_dataset = NMTDataset(config['val_path'], data_cfg, src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab, is_train=False)
    test_dataset = NMTDataset(config['test_path'], data_cfg, src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab, is_train=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config.get('num_workers', 0))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=config.get('num_workers', 0))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.src_vocab, train_dataset.tgt_vocab
