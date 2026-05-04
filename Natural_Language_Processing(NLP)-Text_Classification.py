# NATURAL LANGUAGE PROCESSING(NLP) - Text Classification


#Import Libraries


import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Load Dataset
data = pd.read_csv("Datasets/Sentimentdatasets.csv")  

#Text Preprocessing

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

data['clean_text'] = data['Text'].apply(preprocess)


#Feature Extraction (TF-IDF)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Text'])
y = data['Text']


#Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Train Model

model = LogisticRegression()
model.fit(X_train, y_train)

#Evaluation

y_pred = model.predict(X_test)
print('y_pred')



print(classification_report(y_test, y_pred))
