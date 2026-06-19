# ECGR 4106 Homework 2

This folder contains the completed Homework 2 notebooks and written report for ECGR 4106 - Intro to Deep Learning. The homework studies character-level next-character prediction using recurrent neural networks in PyTorch.

## Files

| File | Description |
|---|---|
| `hw2_problem1.ipynb` | Problem 1 notebook. Trains and compares `nn.RNN`, `nn.LSTM`, and `nn.GRU` on the assigned next-character prediction paragraph using sequence lengths 10, 20, and 30. |
| `hw2_problem2.ipynb` | Problem 2 notebook. Trains and compares LSTM and GRU models on tiny Shakespeare using sequence lengths 20, 30, and 50, plus a hyperparameter study. |

## Environment

The notebooks are designed to run in Google Colab.

Recommended runtime:

- Python 3
- Google Colab GPU runtime
- PyTorch
- NumPy
- pandas
- matplotlib

The notebooks automatically select GPU when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is not available, the notebooks can still run on CPU, but Problem 2 will take longer.

## How To Run

1. Open Google Colab.
2. Upload or open `hw2_problem1.ipynb` and `hw2_problem2.ipynb`.
3. Select a GPU runtime:

   ```text
   Runtime > Change runtime type > Hardware accelerator > GPU
   ```

4. Run each notebook from top to bottom.
5. If the first PyTorch preflight cell repairs the runtime and restarts Colab, run the notebook again from the top after the restart.
6. Review the final summary tables printed at the bottom of each notebook.

## Saved Outputs

When run in Colab, the notebooks save CSV summaries under:

```text
/content/ECGR4106_HW2_outputs
```

Expected CSV outputs include:

- `problem1_summary.csv`
- `problem2_part1_summary.csv`
- `problem2_hyperparameter_summary.csv`
- `problem2_seq50_summary.csv`

These files contain the metrics used in the written report.

## Experiment Overview

### Problem 1

Problem 1 uses the assigned paragraph about next-character prediction. The text is encoded at the character level, then converted into overlapping fixed-length windows. Each model predicts the next character at every time step.

Models compared:

- Vanilla RNN
- LSTM
- GRU

Sequence lengths:

- 10
- 20
- 30

Reported metrics:

- Training loss
- Validation loss
- Validation accuracy
- Validation perplexity
- Training time
- Inference time
- Trainable parameters
- Model size
- Approximate multiply-add complexity per sequence

### Problem 2

Problem 2 uses the tiny Shakespeare dataset. The notebook downloads the dataset automatically and uses a practical subset by default so the full experiment can run in a normal Colab session.

Main comparisons:

- LSTM vs. GRU at sequence lengths 20 and 30
- Hyperparameter changes at sequence length 30
- LSTM vs. GRU at sequence length 50

Hyperparameters explored:

- Fully connected output head
- Number of recurrent layers
- Hidden-state size
- Embedding size
- Dropout

The notebook also prints generated character sequences so the trained models can be compared qualitatively.

## Notes On Metrics

Validation accuracy is measured at the character level on held-out windows.

Validation perplexity is computed from validation cross-entropy:

```text
perplexity = exp(validation loss)
```

Lower validation loss and lower validation perplexity indicate a better language model.

The reported computational complexity is an approximate multiply-add count per sample sequence. It is used for relative comparison across model types and sequence lengths. A vanilla RNN uses one recurrent gate, a GRU uses three gates, and an LSTM uses four gates, so LSTM generally has the largest computation and parameter count.


