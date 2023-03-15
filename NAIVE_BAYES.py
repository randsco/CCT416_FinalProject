import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Input files
FILENAME = "G5-2"

# Load the dataset
tweets = pd.read_csv(f'output/{FILENAME}_labeled.csv')

# Replace any np.nan values with an empty string
tweets = tweets.replace(np.nan, '', regex=True)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(tweets['Cleaned_tweet'], tweets['label'], test_size=0.30, random_state=42)
# commonly used rule of thumb is to split the dataset into 70% for training and 30% for testing



# Convert the tweets into feature vectors
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Print the classification report and confusion matrix
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

with open(f'output/{FILENAME}_classification_report_naive.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))

# Plot a confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix(y_test, y_pred))
ax.grid(False)
plt.title(f'{FILENAME} Confusion Matrix (Naive Bayes)')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('Positive', 'Negative', 'Neutral'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('Positive', 'Negative', 'Neutral'))
for i in range(3):
    for j in range(3):
        ax.text(j, i, confusion_matrix(y_test, y_pred)[i, j], ha='center', va='center', color='white')
plt.show()



# Load the new text data
new_tweets = pd.read_csv(f'output/{FILENAME}_cleaned.csv', usecols=['Tweet', 'Cleaned_tweet'])

import re
import string

# Replace missing values with empty strings
new_tweets['Cleaned_tweet'] = new_tweets['Cleaned_tweet'].fillna('')
new_tweets['Tweet'] = new_tweets['Tweet'].fillna('')

def clean_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove mentions and hashtags
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

# Preprocess the new text data
new_tweets['Cleaned_tweet'] = new_tweets['Tweet'].apply(clean_tweet)

# Convert the new text data into feature vectors using the vectorizer
X_new = vectorizer.transform(new_tweets['Cleaned_tweet'])

# Predict the sentiment of the new text data using the trained Naive Bayes model
y_new_pred = clf.predict(X_new)

# Print the predicted sentiment labels for the new text data
print('Predicted Sentiment Labels for New Text Data:')
print(y_new_pred)

# Add a new column to new_tweets and assign y_new_pred to it
new_tweets['predicted_sentiment'] = y_new_pred

# Get the count of predicted sentiment labels and plot a bar graph
sentiment_counts = new_tweets['predicted_sentiment'].value_counts()
sentiment_counts.plot(kind='bar')
plt.title(f'{FILENAME} Predicted Sentiment Labels (Naive Bayes)')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.show()
