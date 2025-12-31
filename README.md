# IMDB-Sentiment-Analysis
Sentiment analysis on IMDB movie reviews using NLP and machine learning

## 📌 Project Overview
This project performs **sentiment analysis** on IMDB movie reviews using **Natural Language Processing (NLP)** techniques.  
The goal is to classify each movie review as **Positive** or **Negative** by building a complete machine learning pipeline from raw text to model evaluation and visualization.

---

## 🎯 Problem Statement
Movie reviews are unstructured textual data. The objective of this project is to:
- Clean and preprocess raw text reviews
- Convert text data into numerical features
- Train machine learning models to predict sentiment
- Evaluate and compare model performance

---

## 🗂️ Dataset
 **Name**: IMDB Dataset of 50K Movie Reviews
- **Source**: Kaggle
- **Link**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Total Reviews**: 50,000  
- **Classes**:
  - Positive: 25,000  
  - Negative: 25,000  

---

## 🧠 Methodology

### 1️⃣ Text Preprocessing
Raw movie reviews were cleaned using the following steps:
- Removal of HTML tags using regular expressions
- Conversion to lowercase
- Removal of punctuation and numbers
- Stopword removal
- Lemmatization using WordNet

A new column `clean_review` was created after preprocessing.

---

### 2️⃣ Feature Engineering
- Text data was converted into numerical features using **TF-IDF Vectorization**
- Limited to the **top 5000 most important words**
- TF-IDF vectorizer was fitted only on training data to avoid data leakage

---

### 3️⃣ Model Training
Two machine learning models were trained and compared:

#### 🔹 Logistic Regression
- Used as the primary model
- Performs well on high-dimensional sparse text data

#### 🔹 Multinomial Naive Bayes
- Used as a baseline probabilistic classifier
- Fast and effective for text classification

---

### 4️⃣ Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📊 Results

| Model | Accuracy |
|------|----------|
| Logistic Regression | **88.5%** |
| Naive Bayes | 84.9% |

Logistic Regression achieved higher accuracy and was selected as the final model.

---

## 📈 Visualizations
To improve interpretability and presentation, the following visualizations were created:
- Model accuracy comparison bar chart
- Confusion matrix for Logistic Regression
- Confusion matrix for Naive Bayes
- Top positive and negative words influencing sentiment

All plots are available in the `screenshots/` folder.

---

## 🛠️ Tools & Libraries
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib

---

## 🚀 Key Learnings
- Importance of text preprocessing in NLP tasks
- Difference between feature engineering and model learning
- Comparison of probabilistic and discriminative classifiers
- Using visualizations to communicate machine learning results

---

## 🔮 Future Improvements
- Hyperparameter tuning
- N-gram feature extraction
- Deep learning models such as LSTM or BERT

---

## ✅ Conclusion
This project demonstrates an end-to-end NLP workflow for sentiment analysis, covering text preprocessing, feature engineering, model training, evaluation, and visualization. It showcases practical skills relevant to real-world data science and NLP applications.

