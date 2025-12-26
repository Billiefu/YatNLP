# Natural Language Processing [SYSU CSE 2025-1]

> Copyright © 2025 Fu Tszkok

## Repository Description

This repository meticulously curates and preserves the Final Project and its complete code implementation for the **Natural Language Processing** course offered at Sun Yat-sen University during the Fall 2025 semester. The course is instructed by Prof. Xiaojun Quan, whose pedagogy guides students progressively from the foundational concepts of Natural Language Processing to the research of advanced models such as sequence modeling and machine translation, and finally discusses the mechanisms and challenges of contemporary Large Language Models (LLMs).

This project is a complete **Chinese-to-English Neural Machine Translation (NMT) system**. It independently implements an Attention-based Seq2Seq RNN model as well as a Transformer model. The project covers the entire workflow from data preprocessing (tokenization, vocabulary construction), model building, and training/tuning to final evaluation (BLEU/ROUGE metrics and attention visualization). In adherence to copyright, this repository does not contain any lecture slides (PPTs) or other related course materials.

## Copyright Statement

All original code in this repository is licensed under the **[GNU Affero General Public License v3](LICENSE)**, with additional usage restrictions specified in the **[Supplementary Terms](ADDITIONAL_TERMS.md)**. Users must expressly comply with the following conditions:

* **Commercial Use Restriction**
  Any form of commercial use, integration, or distribution requires prior written permission.
* **Academic Citation Requirement**When referenced in research or teaching, proper attribution must include:
  * Original author credit
  * Link to this repository
* **Academic Integrity Clause**
  Prohibits submitting this code (or derivatives) as personal academic work without explicit authorization.

The full legal text is available in the aforementioned license documents. Usage of this repository constitutes acceptance of these terms.

## Repository Structure

The directory structure of this project is organized as follows, designed to maintain modularity and extensibility:

```text
.
│  config.yml        # Global configuration (hyperparameters, paths, model selection)
│  .gitattributes    # Git Large File Storage (LFS) tracking rules
│  pipline.py        # Main entry point (integrates data loading, training, evaluation)
│  train.py          # Training engine (training loops, optimizer, scheduler strategies)
│  evaluate.py       # Evaluation engine (Greedy/Beam Search decoding and metric calculation)
│  dataloader.py     # Data processing (Dataset/DataLoader implementation and vocabulary build)
│  tokenizer.py      # Tokenizer interface (supports Jieba, NLTK, BPE, WordPiece)
│  metric.py         # Metric calculation wrapper (BLEU, ROUGE)
│  utils.py          # Utilities (logging, early stopping, visualization)
│  environment.yml   # Conda environment configuration file
│  Readme.md         # Project documentation
│  LICENSE           # Full text of the GNU Affero General Public License v3.0
│  ADDITIONAL_TERMS.md # Supplementary legal terms regarding academic and commercial usage
│  
├─checkpoints        # Directory for saving model checkpoints
│  └─...             # (e.g., best_model.pt)
│  
├─data               # Directory for datasets and embeddings
│      glove.6B.300d.txt                # English pre-trained embeddings
│      sgns.baidu.bigram-char.300d.txt  # Chinese pre-trained embeddings
│      train_100k.jsonl                 # Large training set
│      train_10k.jsonl                  # Small training set
│      valid.jsonl                      # Validation set
│      test.jsonl                       # Test set
│  
├─logs               # Directory for training logs and output results
│  └─bi-lstm-100k    # Log folder for specific experiments
│      │  *_loss.png        # Loss curve plot
│      │  *_metrics.png     # Metric curve plot
│      │  *_metrics.csv     # Training metric records
│      │  summary.txt       # Experiment summary
│      │  translations.json # Final translation results
│      └─attention_heatmap  # Attention mechanism visualization heatmaps
│  
├─network            # Neural network model definitions
│      lstm.py          # RNN/LSTM model implementation (w/ Attention)
│      transformer.py   # Transformer model implementation (w/ Sparse/GQA)
│  
└─report             # Project report files
    │  report.pdf       # Final PDF report
    │  report.tex       # LaTeX source code
    └─figure            # Figures used in the report
```

## Environment

To run the code in this repository, you need to set up the required environment. Although configuring a Conda environment is not particularly difficult, this repository provides a pre-defined `environment.yml` file located in the root directory to simplify this process.

To configure the environment, ensure that you have Anaconda or Miniconda installed. After installation, switch to the current directory in the command line (cmd) and run the following command:

```shell
conda env create -f environment.yml
```

If there are no issues, the environment will be created successfully. You can then activate the environment by running the following command:

```shell
conda activate YatNLP
```

If you are using PyCharm, VSCode, or other IDEs, you can configure the environment directly within the IDE and run the relevant programs from there.

## Execution Instructions

This project manages the training and testing workflows centrally through `pipline.py`. All hyperparameters and path settings are configured via the `config.yml` file, allowing experimental adjustments without modifying the code.

1. **Modify Configuration**: Modify the `config.yml` file according to your needs (e.g., change `model.name` to `stacked-lstm` or `transformer`).
2. **Start Workflow**:

   * **Train Only**:
     ```shell
     python pipline.py --config config.yml --mode train
     ```
   * **Evaluate Only**:
     Ensure a trained model exists in the `checkpoints` directory.
     ```shell
     python pipline.py --config config.yml --mode evaluate
     ```
   * **Full Workflow (Train & Evaluate)**:
     ```shell
     python pipline.py --config config.yml --mode all
     ```

After execution, translation results, loss curves, and attention heatmaps will be automatically saved in the corresponding experiment folder under the `logs/` directory.

## Acknowledgments

I would like to express my sincere gratitude to **Prof. Xiaojun Quan** for his mentorship in the field of Natural Language Processing. His professional insights and dedicated guidance have provided me with sufficient knowledge in this field and offered great encouragement and help for my future research. I am equally thankful to my fiancée, Ms. **Ma Yujie**, for her unwavering support and quiet encouragement. Natural Language Processing is a subject I consider myself relatively weak in, but I believe it will yield diverse gains in the future multimodal environment. I hope all classmates and colleagues will continue to steadfastly contribute to scientific development.

## Contact & Authorization

For technical inquiries, academic collaboration, or commercial licensing, contact the copyright holder via:

* **Academic Email**: `futk@mail2.sysu.edu.cn`
* **Project Discussions**: [Github Issues](https://github.com/Billiefu/YatNLP/issues)
