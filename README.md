#  Topic Classification using Feedforward Neural Network

This repository contains a complete implementation of a topic classification system using a custom Feedforward Neural Network (FFNN), built from scratch using only Python, NumPy, and basic data processing libraries — no high-level frameworks like PyTorch or TensorFlow.

The model is trained on a subset of the [AG News Corpus](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) and classifies news into three categories: **Politics**, **Sports**, and **Economy**.


##  Features

- **Text Preprocessing**: Unigram tokenization, stopword removal, vocabulary building
- **Input Representation**: Word indices mapped to embedding matrix
- **Model Architecture**:
  - Embedding layer (random init or GloVe)
  - Mean pooling
  - ReLU activation
  - Dropout regularization
  - Softmax output layer
- **Training**:
  - Stochastic Gradient Descent (SGD)
  - Manual backpropagation
  - Early stopping with validation monitoring
- **Hyperparameter Tuning**: Learning rate, embedding size, dropout
- **Evaluation**: Accuracy, Precision, Recall, F1-score

---

##  Performance Summary

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 84.1%     |
| Precision  | 84.3%     |
| Recall     | 84.1%     |
| F1-Score   | 84.0%     |

>  With GloVe embeddings, performance improved slightly in generalization.

---

##  Repository Structure

```
.
├── assignment_acp23pks.ipynb      # Jupyter Notebook with full code and results
├── assignment_acp23pks.pdf        # Exported PDF version of the notebook
├── train.csv                      # Training dataset
├── dev.csv                        # Validation dataset
├── test.csv                       # Test dataset
├── glove/
│   └── README.md                  # Instruction placeholder for GloVe embeddings
└── README.md                      # This file
```

---

##  Pretrained Embeddings (Optional)

To use GloVe embeddings:

1. Download from: [GloVe 840B.300d](https://nlp.stanford.edu/data/glove.840B.300d.zip)
2. Place the `.zip` or extracted `.txt` inside the `glove/` directory.
3. Modify the notebook to load from this path and **freeze** embeddings during training.

>  Avoid pushing the GloVe file to GitHub — it's over 2GB and not allowed.

---

##  Author

**Pranav Kumar Sasikumar**  
[MSc Data Analytics, University of Sheffield]  
Feel free to connect on [LinkedIn](https://www.linkedin.com/) or reach out via GitHub!