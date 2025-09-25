# NLP_hw2# CS5760 Natural Language Processing - Homework 2

## Author: Sai Vardhan Reddy Gummadisani
## ID : 700775046


---

## Overview

This repository contains the programming solutions for **Homework 2** in **CS5760: Natural Language Processing**. It covers two main tasks:

1. **Q5:** Calculating evaluation metrics from a multi-class confusion matrix.
2. **Q8:** Building a simple bigram language model to compute and compare sentence probabilities.

The code is written in **Python** using **NumPy** and standard Python libraries.

---

## What the Code Does

### Q5: Multi-Class Confusion Metrics

1. **Confusion Matrix Setup:**  
   - You created a 3x3 matrix where each row represents the **system’s predicted labels** and each column represents the **true/gold labels**.
   - Example: The first row `[5, 10, 5]` means the model predicted `Cat` 20 times, with 5 correct and 15 misclassified.

2. **Calculating True Positives, Predicted, and Actual Totals:**  
   - `tp` = diagonal of the matrix (correct predictions per class).  
   - `predicted` = sum of each row (total predictions per class).  
   - `actual` = sum of each column (total true instances per class).

3. **Precision and Recall per Class:**  
   - **Precision:** proportion of correct predictions for each class (`tp / predicted`).  
   - **Recall:** proportion of true instances correctly predicted (`tp / actual`).

4. **Macro and Micro Averages:**  
   - **Macro:** average of per-class metrics (treats all classes equally).  
   - **Micro:** considers total true positives and total predictions (weighted by class frequency).

5. **Output:**  
   - Displays per-class precision and recall.  
   - Shows macro and micro averaged precision/recall.

---

### Q8: Bigram Language Model

1. **Corpus Setup:**  
   - Created a small example corpus of sentences, each surrounded by `<s>` (start) and `</s>` (end) tokens.

2. **Counting Unigrams and Bigrams:**  
   - `unigram_counts` = counts of individual words.  
   - `bigram_counts` = counts of consecutive word pairs.

3. **Bigram Probability Function:**  
   - `P(w2 | w1) = count(w1, w2) / count(w1)`  
   - Computes the probability of a word given the previous word.

4. **Sentence Probability:**  
   - For a given sentence, multiplies probabilities of all consecutive bigrams to get the sentence probability.

5. **Comparison of Sentences:**  
   - Calculated probabilities for two example sentences.  
   - Prints which sentence the bigram model prefers (higher probability).

6. **Output Example:**  
   - Shows exact sentence probabilities.  
   - Identifies the preferred sentence according to the model.

---

