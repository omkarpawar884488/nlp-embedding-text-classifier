# NLP Text Classification Pipeline  
*(TF-IDF & Transformer Embedding Experiments)*

This repository contains a complete Natural Language Processing (NLP) text classification pipeline along with a Streamlit application for interactive predictions.

The system predicts whether a given piece of text is:

- **Positive**
- **Negative**
- **Neutral**
- **Irrelevant**

Multiple modeling approaches were implemented and benchmarked, including both classical machine learning techniques and transformer-based sentence embeddings.

---

## 🧠 Project Objective

The goal of this project was to:

- Compare traditional **TF-IDF-based models** with modern **transformer-based embeddings**
- Evaluate performance differences
- Deploy the best-performing model in a lightweight web application

---

## 🧪 Models Implemented

Four different models were trained and evaluated:

| Model | Features Used | Algorithm | Description |
|-------|--------------|-----------|-------------|
| Model-1 | TF-IDF (unigrams + bigrams) | Logistic Regression | Baseline model |
| Model-2 | TF-IDF + Engineered Features | Logistic Regression | Added text statistics (length, punctuation, etc.) |
| Model-3 | Sentence Embeddings (all-MiniLM-L6-v2) | Feedforward Neural Network | Context-aware representation |
| Model-4 | Sentence Embeddings (all-MiniLM-L6-v2) | Logistic Regression | Simpler embedding classifier |

---

## 🏆 Final Model Selection

After evaluation, **Model-1 (TF-IDF + Logistic Regression)** achieved the best overall performance and was selected for deployment.

This highlights an important practical insight:

> For this dataset, a well-tuned classical ML approach outperformed transformer-based embedding models.

The deployed Streamlit app uses this best-performing model.

---

## 🛠 Technical Components

- Text preprocessing and normalization  
- TF-IDF vectorization (unigrams + bigrams)  
- Transformer sentence embeddings (`all-MiniLM-L6-v2`)  
- Logistic Regression classifier  
- Feedforward Neural Network  
- Model evaluation (accuracy, classification report)  
- Streamlit-based UI deployment  


---

## 🗂 Project Structure

```
nlp-embedding-text-classifier/
├── app/
│   └── app.py                  # Streamlit front-end
├── notebooks/
│   └── *.ipynb                 # Exploratory analysis & experiments
├── results/
├── src/
│   ├── preprocessing.py        # Text cleaning logic
│   ├── embedding_pipeline.py   # Embedding generation
│   ├── model_training.py       # Training routines
│   └── evaluation.py           # Evaluation logic
├── .gitignore
├── README.md
└── requirements.txt            # Project dependencies
```

---



## 🚀 How It Works

### 1️⃣ Preprocessing  
Raw text is cleaned and normalized.

### 2️⃣ Feature Extraction  
Either:
- TF-IDF vectorization  
or  
- Transformer-based sentence embeddings  

### 3️⃣ Model Training  
Classifiers are trained to map features → sentiment labels.

### 4️⃣ Evaluation  
Performance is measured on held-out test data.

### 5️⃣ Deployment  
The best-performing model powers the Streamlit application.

---


## 📌 Usage

### Run the Streamlit App

1. Run the app locally:
   ```bash
   streamlit run app/app.py
   ```

2. Type any text in the UI to get a prediction with their probability:
   - **Positive**
   - **Negative**
   - **Neutral** 
   - **Irrelevant**  
   …based on the model’s classifier.

---


This lets you rapidly test and validate model behavior in practice.

---

## 🧪 Notes

- The project does **not include training datasets or large binary artifacts** (e.g., embeddings, models) — those are omitted by design.  
- The model can be retrained using your own dataset following the notebooks and scripts in this repo.

---


## 🧠 Applicability

This project showcases:

✔ Classical NLP vs Transformer comparison

✔ End-to-end ML pipeline development

✔ Model benchmarking and selection

✔ Lightweight ML deployment using Streamlit

✔ Practical experimentation mindset

It can be used as a foundation for:
- Sentiment analysis tools  
- Semantic search systems  
- Text categorization projects  
- Custom NLP applications

---


## 📍 Learn More

Explore the notebooks for details on model training, evaluation methods, and embedding experiments.
