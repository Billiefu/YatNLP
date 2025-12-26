"""
Copyright (C) 2025 Fu Tszkok

:module: tokenizer
:function: Provides unified interface for various tokenization methods (Jieba, NLTK, BPE, WordPiece).
:author: Fu Tszkok
:date: 2025-12-25
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import abc
import os
import re
from typing import List

import jieba
import nltk

# Attempt to import huggingface tokenizers library for advanced statistical tokenization
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
except ImportError:
    print("Warning: 'tokenizers' library not found. BPE and WordPiece will not work.")


class BaseTokenizer(abc.ABC):
    """Abstract Base Class for all tokenizers."""

    @abc.abstractmethod
    def train(self, texts: List[str]):
        """Trains the tokenizer on a list of texts (if applicable)."""
        pass

    @abc.abstractmethod
    def encode(self, text: str) -> List[str]:
        """Tokenizes input text into a list of string tokens."""
        pass

    def save(self, path: str):
        """Saves the tokenizer model to disk."""
        pass

    def load(self, path: str):
        """Loads the tokenizer model from disk."""
        pass


class JiebaTokenizer(BaseTokenizer):
    """Tokenizer wrapper for Jieba (Chinese segmentation)."""

    def train(self, texts: List[str]):
        """Jieba is rule/dictionary based and does not require training."""
        pass

    def encode(self, text: str) -> List[str]:
        """Encodes text using Jieba cut."""
        return list(jieba.cut(text.strip()))


class SimpleRegexTokenizer(BaseTokenizer):
    """Simple tokenizer splitting by space and punctuation."""

    def train(self, texts: List[str]):
        """No training required."""
        pass

    def encode(self, text: str) -> List[str]:
        """Separates punctuation and splits by whitespace."""
        text = text.strip().lower()
        # Add spaces around punctuation to isolate them
        text = re.sub(r'([.,!?;:()])', r' \1 ', text)
        return text.split()


class NLTKCustomTokenizer(BaseTokenizer):
    """Regex-based tokenizer using NLTK patterns."""

    def __init__(self):
        """Initializes regex patterns for acronyms, numbers, and words."""
        base_pattern = r"\w+(?:[-']\w+)*|\S\w*"

        acronyms = r"(?:\w+\.)+\w+(?:\.)*"
        pattern = acronyms + r"|" + base_pattern

        numbers = r"\$?\d+(?:\.\d+)?%?"
        pattern = numbers + r"|" + pattern

        ellipsis = r"\.\.\."
        self.pattern = ellipsis + r"|" + pattern

    def train(self, texts: List[str]):
        """No training required."""
        pass

    def encode(self, text: str) -> List[str]:
        """Tokenizes using NLTK regexp_tokenize."""
        text = text.strip().lower()
        return nltk.tokenize.regexp_tokenize(text, self.pattern)


class HFModelTokenizer(BaseTokenizer):
    """Wrapper for HuggingFace Tokenizers (BPE / WordPiece)."""

    def __init__(self, model_type='bpe', vocab_size=30000, min_frequency=2):
        """Initializes the statistical tokenizer.
        :param model_type: 'bpe' or 'wordpiece'.
        :param vocab_size: Maximum vocabulary size.
        :param min_frequency: Minimum frequency for token inclusion.
        """
        self.tokenizer = None
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.trained = False

    def train(self, texts: List[str]):
        """Trains the tokenizer on the provided corpus.
        :param texts: List of training sentences.
        """
        print(f"Training {self.model_type.upper()} tokenizer on {len(texts)} samples...")

        if self.model_type == 'bpe':
            model = models.BPE(unk_token="<unk>")
            self.tokenizer = Tokenizer(model)
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.BpeTrainer(
                special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency
            )
        elif self.model_type == 'wordpiece':
            model = models.WordPiece(unk_token="<unk>")
            self.tokenizer = Tokenizer(model)
            self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordPieceTrainer(
                special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency
            )
        else:
            raise ValueError("model_type must be 'bpe' or 'wordpiece'")

        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.trained = True

    def encode(self, text: str) -> List[str]:
        """Encodes text using the trained model.
        :return: List of token strings.
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not trained or loaded yet.")
        return self.tokenizer.encode(text).tokens

    def save(self, path: str):
        """Saves the trained tokenizer model to JSON."""
        if self.tokenizer:
            self.tokenizer.save(path)

    def load(self, path: str):
        """Loads the tokenizer model from JSON."""
        if os.path.exists(path):
            self.tokenizer = Tokenizer.from_file(path)
            self.trained = True
        else:
            print(f"Tokenizer model not found at {path}, need training.")


def get_tokenizer(type_name: str, **kwargs):
    """Factory function to get a tokenizer instance.
    :param type_name: Name of tokenizer ('jieba', 'simple', 'nltk', 'bpe', 'wordpiece').
    :param kwargs: Arguments for statistical tokenizers (vocab_size, etc).
    :return: An instance of a BaseTokenizer subclass.
    """
    type_name = type_name.lower()
    if type_name == 'jieba':
        return JiebaTokenizer()
    elif type_name == 'simple':
        return SimpleRegexTokenizer()
    elif type_name == 'nltk':
        return NLTKCustomTokenizer()
    elif type_name == 'bpe':
        return HFModelTokenizer(model_type='bpe', **kwargs)
    elif type_name == 'wordpiece':
        return HFModelTokenizer(model_type='wordpiece', **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {type_name}")
