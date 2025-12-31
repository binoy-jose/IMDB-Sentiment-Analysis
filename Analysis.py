# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 11:46:29 2025

@author: B-Book
"""

import pandas as pd

# loading Data set
df= pd.read_csv(r"E:\Binoy\AI Projects\Project 2\IMDB Dataset.csv")

# Data exploration
df.head()
#df.info()
df['sentiment'].value_counts()
df.isnull().sum()
df['review'][0]

##### Data cleaning #####

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

stop_words = set (stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Creating a text cleaning function

def clean_text(text):
    #1. Remove HTML tags
    text = re.sub(r'<.*?>', '' , text)
    
    #2. Convert to lowercase
    text = text.lower()
    
    #3. Remove punctuations and numbers
    text =re.sub(r'[^a-z\s]', '' , text)
    
    #4. split sentence into words
    words = text.split()
    #5. Remove stopwords and lemmatize
    words=[
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    #6.Join word back into sentence
    return " ".join(words)


df['clean_review']= df['review'].apply(clean_text)
    
#print("ORIGINAL REVIEW:\n")
#print(df['review'][0])

#print("\nCLEANED REVIEW:\n")
#print(df['clean_review'][0])

###### Analysing data   #######

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

x =df['clean_review']
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 42)

#TF-IDF coversion

tfidf = TfidfVectorizer(max_features=5000)

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

#logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#training the model
model = LogisticRegression( max_iter= 1000)
model.fit(x_train_tfidf, y_train)

#prediction
y_pred = model.predict(x_test_tfidf)

#Accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print ('Accuracy:', accuracy)

#Confusion Matrix
#confusion_matrix(y_test, y_pred)

#print (classification_report(y_test, y_pred))
    
#### Naive Bayes Model #####

from sklearn.naive_bayes import MultinomialNB

#Creating the Naive Bayes Model
nb_model = MultinomialNB()

#Train the model
nb_model.fit(x_train_tfidf, y_train)

#predictions
y_pred_nb = nb_model.predict(x_test_tfidf)

#Accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
#print('Naive Bayes Accuracy:', accuracy_nb)

#Classification report
#print(classification_report(y_test, y_pred_nb))

###### Charts #####

import matplotlib.pyplot as plt

#Accuracy Comparison Chart
models = ['Logistic Regression', 'Naive Bayes']
accuracies = [0.8851, 0.8495]

plt.figure()
plt.bar(models, accuracies)
plt.ylim (0.8,0.9)
plt.title ('Model Accuracy Comparison')
plt.ylabel ('Accuracy')
plt.xlabel ('Model')

for i, acc in enumerate(accuracies):
    plt.text(i,acc + 0.002, f"{acc:.2f}", ha = 'center')

plt.show()

#Confusion Matrix Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)

plt.figure()
plt.imshow(cm_nb)
plt.title("Confusion Matrix - Naive Bayes")
plt.colorbar()

plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_nb[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Confusion Logistic regression
cm_lr = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm_lr)
plt.title("Confusion Matrix - Logistic Regression")
plt.colorbar()

plt.xticks([0, 1], ['Negative', 'Positive'])
plt.yticks([0, 1], ['Negative', 'Positive'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_lr[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Top TF-IDF words from Logistic Regression
import numpy as np

feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

top_positive = np.argsort(coefficients)[-10:]
top_negative = np.argsort(coefficients)[:10]

plt.figure()
plt.barh(feature_names[top_positive], coefficients[top_positive])
plt.title("Top Positive Words")
plt.xlabel("Weight")
plt.show()

plt.figure()
plt.barh(feature_names[top_negative], coefficients[top_negative])
plt.title("Top Negative Words")
plt.xlabel("Weight")
plt.show()






    
    
    
    
    
    
    
    
    
    