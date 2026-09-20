# Stacked ML vs DistilBERT

| Model | Accuracy | Precision | Recall | F1 | Macro F1 | Test size | Train time |
|---|---|---|---|---|---|---|---|
| Stacked ML - slide 5 (reported) | 80.3% | 0.80 | 0.79 | 0.79 | - | - | - |
| DistilBERT - slide 5 (reported) | 92.7% | 0.93 | 0.92 | 0.92 | - | - | - |
| Stacked ML - corrected | 78.99% | 0.782 | 0.774 | 0.778 | 0.789 | 5701 | 543 s |
| DistilBERT - fine-tuned | 92.55% | 0.914 | 0.931 | 0.922 | 0.925 | 5701 | 169 s |

Precision / recall / F1 are for the sarcastic class (as on slide 5).
*The Colab replica fits TF-IDF on all data before splitting and keeps duplicates, and uses a different test split; it is only there to check the reported 80.3%.

Same 5701 test headlines for both corrected runs.
Accuracy gap (DistilBERT - Stacked ML): 13.6 percentage points (slide 5 reports 12.4).
Errors: Stacked ML 1198, DistilBERT 425 -> DistilBERT makes 65% fewer mistakes.
