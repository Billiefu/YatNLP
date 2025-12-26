"""
Copyright (C) 2025 Fu Tszkok

:module: metric
:function: Encapsulates BLEU and ROUGE metric calculations using standard libraries (sacrebleu, rouge_score).
:author: Fu Tszkok
:date: 2025-12-20
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import re
from typing import List

import sacrebleu
from rouge_score import rouge_scorer


class BleuCalculator:
    """Helper class to compute BLEU scores."""

    def __init__(self, tokenize='13a'):
        """Initializes BLEU calculator.
        :param tokenize: Tokenization method used by sacrebleu (default '13a').
        """
        self.tokenize = tokenize

    def compute_bleu(self, hypotheses: List[str], references: List[str]) -> float:
        """Computes corpus-level BLEU score.
        :param hypotheses: List of generated sentences.
        :param references: List of reference sentences.
        :return: BLEU score (0-100).
        """
        if not hypotheses or not references: return 0.0
        # Basic detokenization/cleaning for consistency
        references = [re.sub(r'\s+([.,!?;:()])', r'\1', ref) for ref in references]
        hypotheses = [re.sub(r'\s+([.,!?;:()])', r'\1', hyp) for hyp in hypotheses]

        # sacrebleu expects list of reference lists
        refs = [references]
        bleu = sacrebleu.corpus_bleu(hypotheses, refs, tokenize=self.tokenize)
        return bleu.score

    def compute_sentence_bleu(self, hypothesis: str, reference: str) -> float:
        """Computes sentence-level BLEU score.
        :param hypothesis: Single generated sentence.
        :param reference: Single reference sentence.
        :return: BLEU score.
        """
        return sacrebleu.sentence_bleu(hypothesis, [reference], tokenize=self.tokenize).score


class RougeCalculator:
    """Helper class to compute ROUGE scores."""

    def __init__(self, rouge_type='rougeL'):
        """Initializes ROUGE calculator.
        :param rouge_type: Metric type (default 'rougeL' for Longest Common Subsequence).
        """
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        self.rouge_type = rouge_type

    def compute_corpus_rouge(self, hypotheses: List[str], references: List[str]) -> float:
        """Computes average corpus-level ROUGE score.
        :param hypotheses: List of generated sentences.
        :param references: List of reference sentences.
        :return: Average ROUGE F-measure (0-100).
        """
        if not hypotheses or not references: return 0.0
        total_fmeasure = 0.0
        for hyp, ref in zip(hypotheses, references):
            scores = self.scorer.score(ref, hyp)
            total_fmeasure += scores[self.rouge_type].fmeasure
        return (total_fmeasure / len(hypotheses)) * 100

    def compute_sentence_rouge(self, hypothesis: str, reference: str) -> float:
        """Computes sentence-level ROUGE score.
        :param hypothesis: Single generated sentence.
        :param reference: Single reference sentence.
        :return: ROUGE F-measure (0-100).
        """
        scores = self.scorer.score(reference, hypothesis)
        return scores[self.rouge_type].fmeasure * 100
