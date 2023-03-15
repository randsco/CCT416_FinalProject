import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

# Input files
FILENAME = "G5-2"

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Open the CSV file and read the tweets
with open(f'output/{FILENAME}_cleaned.csv', 'r') as file:
    reader = csv.DictReader(file)
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    pos_score_total = 0
    neg_score_total = 0
    neu_score_total = 0

    for row in reader:
        tweet = row['Cleaned_tweet']

        # Use the VADER sentiment analyzer to compute the sentiment score
        score = analyzer.polarity_scores(tweet)['compound']

        # Categorize the sentiment score as positive, negative, or neutral
        if score > 0:
            positive_count += 1
        elif score < 0:
            negative_count += 1
        else:
            neutral_count += 1

        # Add the VADER scores to the total for computing the average later
        pos_score_total += analyzer.polarity_scores(tweet)['pos']
        neg_score_total += analyzer.polarity_scores(tweet)['neg']
        neu_score_total += analyzer.polarity_scores(tweet)['neu']

    # Compute the average VADER scores
    total_count = positive_count + negative_count + neutral_count
    if total_count > 0:
        pos_score_avg = pos_score_total / total_count
        neg_score_avg = neg_score_total / total_count
        neu_score_avg = neu_score_total / total_count
    else:
        pos_score_avg = neg_score_avg = neu_score_avg = 0.0

    # Plot the sentiment summary as a bar chart
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [positive_count, negative_count, neutral_count]
    plt.bar(labels, counts)
    plt.xlabel('Sentiment')
    plt.ylabel('Number of tweets')
    plt.title(f'{FILENAME} Sentiment Analysis (VADER)')
    plt.show() # Display the chart

    # Write the sentiment summary and VADER statistics to a new CSV file
    with open(f'output/{FILENAME}_vader_sentiment_summary.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        writer.writerow(['Sentiment', 'Count'])
        writer.writerow(['Positive', positive_count])
        writer.writerow(['Negative', negative_count])
        writer.writerow(['Neutral', neutral_count])

        writer.writerow([])  # Add a blank line between sections

        writer.writerow(['VADER Statistic', 'Value'])
        writer.writerow(['Positive Score (avg)', pos_score_avg])
        writer.writerow(['Negative Score (avg)', neg_score_avg])
        writer.writerow(['Neutral Score (avg)', neu_score_avg])


print('Average Positive Score:', pos_score_avg)
print('Average Negative Score:', neg_score_avg)
print('Average Neutral Score:', neu_score_avg)
