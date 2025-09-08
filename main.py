import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

np.random.seed(42)

df = pd.read_csv('IMDB Dataset.csv')

print(df.head())

# Clean the dataframe - typically the spam dataset has extra columns we don't need
# Keep only the relevant columns (usually the first two)
df.columns = ['review', 'sentiment']

#Check for Missing values
print('Missing Values: ')
print(df.isnull().sum())

df = df.dropna()

#Initialize Stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(review):
    #Put everything in lowercase
    review = review.lower()

    #Remove every character not in the alphabet
    review = re.sub(r'[^a-zA-Z\s]', '', review)

    #Tokenize
    words = review.split(' ')

    #Remove Stopwords
    preprocess_words = [stemmer.stem(word) for word in words if word not in stop_words]

    #Join the words back together
    return ' '.join(preprocess_words)

#Apply preprocessing to all messages
print('Preprocessing messages...')
df['Processed_review'] = df['review'].apply(preprocess_text)

#To Print the Processed Reviews
""" for i in range(1):
    print(f'Original: {df['review'].iloc[i]}')
    print(f'\nProcessed: {df['Processed_review'].iloc[i]}')
    print("---") """

df['sentiment_num'] = df['sentiment'].map({'negative': 0, 'positive': 1})

X_train, X_test, y_train, y_test = train_test_split(df['Processed_review'], df['sentiment_num'], test_size=0.2, random_state=42, stratify=df['sentiment_num'])

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=7500)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

""" NAIVES BAYES MODEL """
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

y_pred_nb = nb_classifier.predict(X_test_tfidf)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"\nThe accuracy of the Naives Bayes Model is {accuracy_nb:.4f}")
print('\nClassification Report')
print(classification_report(y_test, y_pred_nb, target_names=['negative', 'positive']))


""" LOGISTIC REGRESSION """
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)

y_pred_lr = lr_classifier.predict(X_test_tfidf)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f'\nThe accuracy of Logistic Regression Model is {accuracy_lr}')
print('\nClassification Report')
print(classification_report(y_test, y_pred_lr, target_names=['negative', 'positive']))

cm_lr = confusion_matrix(y_test, y_pred_lr, )
cm_nb = confusion_matrix(y_test, y_pred_nb, )

fig, axes = plt.subplots(1, 2, figsize= (15, 5))

# Naive Bayes confusion matrix
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Naive Bayes Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['negative', 'positive'])
axes[0].set_yticklabels(['negative', 'positive'])

# Logistic Regression confusion matrix
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['negative', 'positive'])
axes[1].set_yticklabels(['negative', 'positive'])

plt.tight_layout()
plt.show()

# Compare accuracies
print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")

feature_names = tfidf.get_feature_names_out()
coefficients = lr_classifier.coef_[0]

features_importance =pd.DataFrame({
    'features': feature_names,
    'importance': coefficients
})

feature_importance = features_importance.sort_values('importance', ascending=False)

#Top words for Positives
print("\nTop 20 features indicating Positive Reviews:")
print(feature_importance.head(20))

#Top words for Negatives
print("\nTop 20 features indicating Negative Reviews:")
print(feature_importance.tail(20))