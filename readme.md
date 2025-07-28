# Naive Bayes Word Sense Disambiguation

A simple Naive Bayes classifier for word‐sense disambiguation (WSD) using **labelled** training data. Given a target word and its context, the model learns sense distributions from a training corpus and predicts the most likely sense in unseen examples.

---

## Features

- **Lemmatisation** and **stopword filtering**
- Handles XML‑style sense tags and sentence UIDs
- Laplace (add‑one) smoothing for robust probability estimation
- Computes and logs **prior** and **conditional** probabilities
- Reports overall classification accuracy against a gold standard

---

## Prerequisites

- **Python 3.7+**
- **NLTK** data packages:
  ```bash
  pip install nltk
  python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
  ```

---

## Project Structure

```
.
├── README.md             ← this file
├── NB.py                 ← main classifier implementation
├── DICT/                 ← sense‑dictionary files (*.dic)
│   └── <word>.dic
├── TRAIN/                ← training corpora files (*.cor)
│   └── <word>.cor
├── TEST/                 ← test examples (*.eval)
│   └── <word>[-p].eval
└── GOLD/                 ← gold‑standard sense assignments
    └── <word>[-p]
```

---

## Usage

```bash
python NB.py <target_word> [<eval_flag>]
```

- `<target_word>`: the word to disambiguate (must match files in `DICT/`, `TRAIN/`, etc.).
- `<eval_flag>`: optional suffix `-p` or other flag matching your `TEST/` and `GOLD/` filenames. Default is `-p`.

### Examples

- Disambiguate senses of **“bank”** in default mode:
  ```bash
  python NB.py bank -p
  ```
- Run on test file `TEST/bank.eval`:
  ```bash
  python NB.py bank
  ```

After running, you’ll see:

```
Working with:  bank -p
Reading dictionary...
…Done!
Reading training data…
…Done!
Reading Test data…
…Done!
Reading GOLD…
…Done!
Training…
…Done!
Testing…
    ***********************************
    Success rate:  0.82
    ***********************************
Well done!
```

---

## How It Works

1. **Text Scrubbing**
   - Lowercases, strips punctuation, removes NLTK stopwords, and lemmatises each token.
2. **Dictionary Parsing** (`readDict`)
   - Reads `<word>.dic`, extracts sentence UIDs and their gold sense tags.
3. **Training** (`readTrain`)
   - Iterates through the `.cor` file, collects sense instance counts (`countInst`) and conditional word counts per sense (`countCond`), with add‑one smoothing.
4. **Probability Computation** (`calcProbs`)
   - Computes log priors (`logPrior[s] = log(countInst[s] / N)`) and log conditionals (`logCond[s][w] = log(countCond[s][w] / total_w)`).
5. **Prediction** (`testNaiveBayes`)
   - For each test sentence, sums log‑probabilities over words and picks the sense with maximum posterior probability.
6. **Evaluation**
   - Compares predicted sense UIDs (mapped via `tagUID`) against gold standard in `GOLD/` and computes overall accuracy.

---

## Configuration & Data

- Place your sense dictionary files in `DICT/`, training files in `TRAIN/`, test files in `TEST/`, and gold‑standard files in `GOLD/`.
- Filenames must follow `<word>.<ext>` conventions.
- To add a new target word, simply add corresponding `.dic`, `.cor`, `.eval`, and gold files, then rerun the script.

