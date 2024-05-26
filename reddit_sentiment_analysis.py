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
dz = pd.read_csv('data/Copy of Reddit_Data.csv')
dz.head()

# Data preprocessing
dz.dropna(inplace=True)

# Convert to numpy arrays
x = dz.clean_text.to_numpy()
y = dz.category.to_numpy()

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

# Vectorization
tfidf = TfidfVectorizer()
x_train_vect = tfidf.fit_transform(x_train)
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
