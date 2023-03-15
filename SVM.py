import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load cleaned tweet data from a CSV file
df = pd.read_csv('labeled_data.csv')

# Replace any np.nan values with an empty string
tweets = df['Cleaned_tweet'].replace(np.nan, '', regex=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweets, df['label'], test_size=0.3, random_state=42)

# Convert the text data to numerical features using a count vectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict the sentiment of the test data
y_pred = svm.predict(X_test)

# Print the classification report and confusion matrix
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

with open('classification_report_svm.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))

# Save the classification report to a text file
with open('classification_report_svm.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))


# Plot a confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix(y_test, y_pred))
ax.grid(False)
plt.title('Confusion Matrix (SVM)')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('Negative', 'Neutral', 'Positive'))
ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('Negative', 'Neutral', 'Positive'))
for i in range(3):
    for j in range(3):
        ax.text(j, i, confusion_matrix(y_test, y_pred)[i, j], ha='center', va='center', color='white')
plt.show()

# Save the predicted sentiment of the tweets to a new CSV file
df_test = pd.DataFrame({'text': X_test, 'sentiment': y_pred})
df_test.to_csv('svm.csv', index=False)

from sklearn.metrics import accuracy_score

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)



# Load the new text data
new_df = pd.read_csv('tweets-step2.csv')

# Define function to clean tweet data
def clean_tweet(tweet):
    # Code to clean tweet data goes here
    return tweet

new_tweets = new_df['Tweet'].replace(np.nan, '', regex=True)
new_tweets = new_tweets.apply(clean_tweet)

new_X = vectorizer.transform(new_tweets)
new_y_pred = svm.predict(new_X)
new_df['sentiment'] = new_y_pred
new_df.to_csv('tweets_predicted.csv', index=False)
sentiments, counts = np.unique(new_y_pred, return_counts=True)
fig, ax = plt.subplots()
ax.bar(sentiments, counts)
ax.set_xticks(sentiments)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel('Count')
ax.set_title('Overall Sentiment (SVM)')
plt.show()

