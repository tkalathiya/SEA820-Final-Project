# SEA820-Final-Project: AI vs Human Text Classifier

This project investigates the growing challenge of distinguishing between **human-written** and **AI-generated** text. Using a combination of traditional machine learning and transformer-based NLP models, we compare performance in identifying the source of a given text excerpt.

---

## Dataset

- **Source**: [AI_Human.csv from Kaggle](https://www.kaggle.com/datasets/)
- **Classes**:
  - `0` → Human-written
  - `1` → AI-generated
- Includes essays, paragraphs, and various writing styles.

---

## Models Used

### Baseline Model (Week 1)
- **Preprocessing**: Text cleaning, tokenization, stopword removal
- **Features**: TF-IDF with unigram + bigram (`max_features=5000`)
- **Classifier**: Logistic Regression
- **Performance**: F1 Score ≈ `0.99`

### Transformer Model (Week 2+)
- **Model**: `DistilBERT-base-uncased`
- **Library**: Hugging Face `transformers` and `datasets`
- **Training**:
  - `epochs=3`
  - `batch_size=16`
  - `learning_rate=2e-5`
- **Performance**: F1 Score ≈ `0.985`

---

## Results Comparison

| Metric     | Logistic Regression | DistilBERT |
|------------|---------------------|------------|
| Accuracy   | 0.99                | 0.985      |
| Precision  | 0.99–1.00           | 0.98       |
| Recall     | 0.99–1.00           | 0.99       |
| F1 Score   | 0.99                | 0.985      |

> Classic ML surprisingly outperforms DistilBERT slightly due to dataset simplicity.


