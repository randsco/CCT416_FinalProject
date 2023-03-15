import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from collections import Counter
import re

# Input files
FILENAME = "G5-2"

# load the csv file into a pandas dataframe
df = pd.read_csv(f'output/{FILENAME}_preprocessed.csv')

# initialize the NLTK lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Replace missing values with empty strings
df['Tweet'] = df['Tweet'].fillna('')

# define a function to clean each tweet
def clean_tweet(tweet):
    # Remove single/2 letters
    tweet = re.sub(r"\b\w\b", "", tweet)
    tweet = re.sub(r"\b\w{1,2}\b", "", tweet)

    # Remove HTML tags
    tweet = re.sub(r"&\w+;", "", tweet)

    # Remove "..."
    tweet = re.sub(r"\.{3}", "", tweet)

    # Remove "" "
    tweet = re.sub(r"\"{1}", "", tweet)

    # tokenize the tweet
    tokens = word_tokenize(tweet.lower())

    # remove digits and punctuation
    tokens = [token for token in tokens if not token.isdigit() and token not in string.punctuation]

    # remove stop words
    tokens = [token for token in tokens if token not in stop_words]

    # lemmatize the tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # join the tokens back into a string
    cleaned_tweet = ' '.join(tokens)

    return cleaned_tweet

# apply the clean_tweet function to each tweet in the dataframe
df['Cleaned_tweet'] = df['Tweet'].apply(clean_tweet)

# save the cleaned tweets to a new CSV file
df.to_csv(f'output/{FILENAME}_cleaned.csv', index=False)

# load the cleaned tweets CSV file into a pandas dataframe
df2 = pd.read_csv(f'output/{FILENAME}_cleaned.csv', usecols=['Cleaned_tweet'])
print(df2['Cleaned_tweet'])

# Replace missing values with empty strings
df2['Cleaned_tweet'] = df2['Cleaned_tweet'].fillna('')

# count the frequency of each word
word_freq = Counter()
for tweet in df2['Cleaned_tweet']:
    word_freq.update(tweet.split())

# get the top 10 most frequent words and their counts
top_words = word_freq.most_common(10)
top_words_list = [word[0] for word in top_words]
top_words_counts = [word[1] for word in top_words]

# plot the top 10 most frequent words on a bar graph
plt.bar(top_words_list, top_words_counts)
plt.title(f'{FILENAME} Top 10 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
