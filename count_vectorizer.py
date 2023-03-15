import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Input files
FILENAME = "G5-2"

# Load the cleaned tweets from a CSV file
df = pd.read_csv(f'output/{FILENAME}_cleaned.csv')

# Replace missing values with empty strings
df['Cleaned_tweet'] = df['Cleaned_tweet'].fillna('')

# Use CountVectorizer to prepare the tweets for a machine learning algorithm
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Cleaned_tweet'])

# Print some information about the feature matrix
print('Shape of feature matrix:', X.shape)
print('Number of unique words:', len(vectorizer.vocabulary_))

# Save the feature matrix to a new CSV file
df_features = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
df_features.to_csv(f'output/{FILENAME}_features.csv', index=False)

# Visualize the 10 most common words in the corpus
word_counts = X.sum(axis=0)
word_counts_df = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'count': word_counts.tolist()[0]
})
word_counts_df = word_counts_df.sort_values(by='count', ascending=False).head(10)

# Top 10 words

plt.bar(word_counts_df['word'], word_counts_df['count'])
plt.title('Top 10 Words in Tweets (CountVectorizer)')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()
