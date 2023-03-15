import csv
from textblob import TextBlob
import matplotlib.pyplot as plt

# Input files
FILENAME = "G5-2"

# Open the CSV file containing tweets
with open(f'output/{FILENAME}_cleaned.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # Initialize variables to keep track of sentiment counts
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Loop through each row in the CSV file
    for row in reader:
        tweet = row[6]
        # Create a TextBlob object for each tweet
        blob = TextBlob(tweet)
        # Get the sentiment polarity of the tweet (ranging from -1 to 1)
        sentiment = blob.sentiment.polarity

        # Classify sentiment as positive, negative or neutral
        if sentiment > 0:
            sentiment_class = 'Positive'
            positive_count += 1
        elif sentiment < 0:
            sentiment_class = 'Negative'
            negative_count += 1
        else:
            sentiment_class = 'Neutral'
            neutral_count += 1

       
    # Plot a bar chart of the overall sentiment
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [positive_count, negative_count, neutral_count]
    colors = 'blue'
    plt.bar(labels, counts)
    plt.title(f'{FILENAME} Sentiment Analysis (TextBlob)')
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    plt.show()

 # Print overall sentiment scores
    print("Positive score:", positive_count)
    print("Negative score:", negative_count)
    print("Neutral score:", neutral_count)