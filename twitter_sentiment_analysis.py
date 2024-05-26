import nltk, warnings, string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/Copy of Twitter_Data.csv')
df.head()

# Data preprocessing
df.dropna(inplace=True)
df.category.replace([-1.0, 0.0, 1.0], ['Negative', 'Neutral', 'Positive'], inplace=True)

# Convert to numpy arrays
x = df.clean_text.to_numpy()
y = df.category.to_numpy()

# Split data
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Vectorization
tfidf = TfidfVectorizer()
x_train_vect = tfidf.fit_transform(x_train)
x_val_vect = tfidf.transform(x_val)
x_test_vect = tfidf.transform(x_test)

print(x_train_vect.shape)

# KNN classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_vect, y_train)

# Predictions
knn_pred = knn.predict(x_test_vect)
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
accuracy = accuracy_score(y_test, knn_pred)
print('Accuracy:', accuracy)
