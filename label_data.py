import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Input files
FILENAME = "G5-2"

# Load the preprocessed data from a CSV file
data = pd.read_csv(f'output/{FILENAME}_cleaned.csv')

# Replace missing values with empty strings
data['Cleaned_tweet'] = data['Cleaned_tweet'].fillna('')

# Create a new column to store the labels
data['label'] = ''

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Define a labeling function that uses VADER sentiment analysis
def label_tweet(tweet):
    # Use VADER sentiment analyzer to classify the tweet as positive, negative or neutral
    scores = analyzer.polarity_scores(tweet)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        label = 'positive'
    elif compound_score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return label


# Iterate over each row in the DataFrame and assign a label to each tweet
for index, row in data.iterrows():

    tweet = row['Cleaned_tweet']
    print(tweet)
    label = label_tweet(tweet)
    data.at[index, 'label'] = label

# Save the labeled data to a new CSV file
data.to_csv(f'output/{FILENAME}_labeled.csv', index=False)
