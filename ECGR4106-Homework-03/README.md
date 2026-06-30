# ECGR 4106 Homework 3

## Overview

This folder contains the solution files for ECGR 4106 Homework 3: Sequence-to-Sequence Machine Translation. The assignment trains GRU encoder-decoder models on the provided `vast_english_french.txt` dataset.

The work is organized into three Google Colab-compatible notebooks:

| File | Description |
|---|---|
| `hw3-problem-1.ipynb` | Question 1 baseline GRU encoder-decoder for English-to-French translation. |
| `hw3-problem-2.ipynb` | Question 2 English-to-French GRU encoder-decoder with Luong attention. |
| `hw3-problem-3.ipynb` | Question 3 French-to-English baseline and attention GRU models. |

## Dataset and Split

All notebooks use `vast_english_french.txt`.

The dataset contains 555 English-French sentence pairs. A fixed random seed of 4106 was used to create one reproducible 80/20 split:

| Split | Count |
|---|---:|
| Training | 444 |
| Validation | 111 |

The split is saved in `hw3_split_indices.json`, and all three notebooks reuse those indices.

## Environment

The notebooks were designed for Google Colab with GPU acceleration.

Main dependencies:

- PyTorch
- nltk
- matplotlib
- pandas
- numpy
- tqdm

Each notebook uses:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Results

| Question | Direction | Model | Sequence Accuracy | BLEU-4 | Final Train CE | Final Val CE |
|---|---|---|---:|---:|---:|---:|
| Q1 | English -> French | Baseline GRU | 0.90% | 0.0667 | 1.5236 | 5.6666 |
| Q2 | English -> French | Attention GRU | 0.90% | 0.1239 | 0.0885 | 5.8977 |
| Q3 | French -> English | Baseline GRU | 0.00% | 0.0944 | 1.4040 | 5.1852 |
| Q3 | French -> English | Attention GRU | 0.00% | 0.1146 | 0.7755 | 5.4519 |

The attention models improved BLEU-4 in both translation directions. Exact sequence accuracy remained low because it requires a complete word-for-word sentence match.

## Saved Outputs

The notebooks save generated files in `outputs/`:

- Loss curve PNG files
- Attention map PNG files
- Metrics JSON files
- Sample prediction CSV files
- PyTorch model checkpoints

Important files include:

| File | Description |
|---|---|
| `outputs/q1_baseline_en_fr_metrics.json` | Question 1 final metrics. |
| `outputs/q2_attention_en_fr_metrics.json` | Question 2 final metrics. |
| `outputs/q3_fr_en_metrics.json` | Question 3 final metrics for both models. |
| `outputs/q1_baseline_en_fr_examples.csv` | Question 1 validation predictions. |
| `outputs/q2_attention_en_fr_examples.csv` | Question 2 validation predictions. |
| `outputs/q3_baseline_fr_en_examples.csv` | Question 3 baseline validation predictions. |
| `outputs/q3_attention_fr_en_examples.csv` | Question 3 attention validation predictions. |

## Running the Notebooks

1. Upload the three notebooks, `vast_english_french.txt`, and `hw3_split_indices.json` to the same Google Drive folder.
2. Open each notebook in Google Colab.
3. Select a GPU runtime.
4. Run the notebooks from top to bottom in order: Question 1, Question 2, then Question 3.
5. Confirm that the outputs are saved in the `outputs/` folder.
