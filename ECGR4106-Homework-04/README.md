# ECGR 4106 Homework 4

This repository contains the ECGR 4106 Homework 4 solution package. The assignment studies Transformer models for character-level language modeling and sequence-to-sequence translation, then compares the Transformer results against prior RNN-based homework results.

The work is organized around four problems:

1. Train a character-level Transformer on the assigned next-character prediction paragraph using sequence lengths 10, 20, and 30.
2. Train and tune a character-level Transformer on Tiny Shakespeare, including sequence lengths 20, 30, and 50 plus a block/head architecture sweep.
3. Build a Transformer encoder-decoder for English-to-French translation using the same 80/20 train-validation split from Homework 3.
4. Repeat the translation experiment in the French-to-English direction and compare which direction is easier for the Transformer to optimize.

The notebooks are written for Google Colab, use PyTorch, automatically select GPU when available, and save metrics, plots, and qualitative generated examples.

## Contents

- `Problem_1_Transformer_Character_Level.ipynb`
- `Problem_2_Tiny_Shakespeare_Transformer.ipynb`
- `Problem_3_English_to_French_Transformer.ipynb`
- `Problem_4_French_to_English_Transformer.ipynb`
- `data/vast_english_french.txt`
- `data/hw3_split_indices.json`

## Datasets

Problems 1 and 2 use character-level text modeling:

- Problem 1 embeds the assigned next-character prediction paragraph directly in the notebook.
- Problem 2 downloads Tiny Shakespeare automatically from the public Karpathy character-RNN dataset source.

Problems 3 and 4 use:

- `data/vast_english_french.txt`
- `data/hw3_split_indices.json`

The split file preserves the same Homework 3 split: 444 training sentence pairs and 111 validation sentence pairs from 555 total pairs. Validation metrics are computed only on the held-out validation set.

## Requirements

The notebooks are designed for Google Colab with GPU enabled:

- Python 3
- PyTorch
- pandas
- matplotlib
- nltk
- tqdm

## Running in Google Colab

Open each notebook in Colab and run top to bottom with GPU enabled. The notebooks install/import the required helper packages, select `cuda` automatically when available, and save report-ready results into `outputs/`.

The Homework 4 PDF says the block/head sweep has eight configurations, but the specified values `blocks = [1, 2, 4]` and `heads = [2, 4]` form six unique combinations. The notebooks run all specified combinations exactly.



