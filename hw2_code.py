"""
CS5760 Natural Language Processing
Homework 2 – Programming Parts (Q5 and Q8)
Author: Sai Vardhan Reddy Gummadisani
ID : 
"""

import numpy as np
from collections import Counter

# ---------- Q5 : Evaluation Metrics from a Multi-Class Confusion Matrix ----------

def confusion_metrics():
    # Confusion matrix: rows = system predictions, cols = gold labels
    cm = np.array([
        [5, 10, 5],   # Predicted Cat
        [15,20,10],   # Predicted Dog
        [0, 15,10]    # Predicted Rabbit
    ], dtype=float)

    labels = ["Cat","Dog","Rabbit"]

    tp = np.diag(cm)
    predicted = cm.sum(axis=1)  # row sums
    actual = cm.sum(axis=0)     # column sums

    precision = tp / predicted
    recall = tp / actual

    macro_precision = precision.mean()
    macro_recall = recall.mean()

    micro_precision = tp.sum() / cm.sum()
    micro_recall = micro_precision

    print("=== Q5: Per-class metrics ===")
    for i, lab in enumerate(labels):
        print(f"{lab}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}")
    print("\nMacro Precision =", round(macro_precision,4))
    print("Macro Recall    =", round(macro_recall,4))
    print("Micro Precision =", round(micro_precision,4))
    print("Micro Recall    =", round(micro_recall,4))
    print("="*40,"\n")


# ---------- Q8 : Bigram Language Model Implementation ----------

def bigram_language_model():
    corpus = [
        ["<s>", "I", "love", "NLP", "</s>"],
        ["<s>", "I", "love", "deep", "learning", "</s>"],
        ["<s>", "deep", "learning", "is", "fun", "</s>"]
    ]

    # Count unigrams and bigrams
    unigram_counts = Counter()
    bigram_counts = Counter()
    for sent in corpus:
        unigram_counts.update(sent)
        bigram_counts.update(zip(sent[:-1], sent[1:]))

    def bigram_prob(w1, w2):
        return bigram_counts[(w1, w2)] / unigram_counts[w1]

    def sentence_prob(sentence):
        p = 1.0
        for w1, w2 in zip(sentence[:-1], sentence[1:]):
            p *= bigram_prob(w1, w2)
        return p

    s1 = ["<s>", "I", "love", "NLP", "</s>"]
    s2 = ["<s>", "I", "love", "deep", "learning", "</s>"]

    p1 = sentence_prob(s1)
    p2 = sentence_prob(s2)

    print("=== Q8: Bigram Language Model ===")
    print("P(<s> I love NLP </s>) =", p1)
    print("P(<s> I love deep learning </s>) =", p2)
    preferred = "Sentence 1" if p1 > p2 else "Sentence 2"
    print("Model prefers:", preferred)
    print("="*40,"\n")


if __name__ == "__main__":
    confusion_metrics()
    bigram_language_model()
