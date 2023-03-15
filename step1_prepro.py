import csv
import preprocessor as p
import re
from collections import Counter

# Input files
FILENAME = "G5-2"

# read the csv file
with open(f"input/{FILENAME}.csv", "r", encoding="UTF-8") as file:
    reader = csv.reader(file)
    data = list(reader)

# print the number of original tweets
print("Number of original tweets:", len(data))

# remove the header row
header = data[0]
data = data[1:]

# initialize the cleaned tweets set and list
seen_tweets = set()
cleaned_tweets = []

# iterate through each row of the data
for row in data:
    # extract the tweet
    tweet = row[3]

    # clean the tweet and remove numbers
    cleaned_tweet = re.sub(r'\d+', '', p.clean(tweet))

    # check if the tweet is a retweet
    if not tweet.startswith("RT "):
        # check if we've seen this tweet before
        if cleaned_tweet not in seen_tweets:
            # mark this tweet as seen
            seen_tweets.add(cleaned_tweet)

            # append the cleaned tweet to the cleaned_tweets list
            cleaned_tweets.append(cleaned_tweet)

# write the cleaned tweets to a new csv file
with open(f"output/{FILENAME}_preprocessed.csv", "w", newline="", encoding="UTF-8") as file:
    writer = csv.writer(file)

    # write the header
    writer.writerow(header)

    # write the cleaned tweets
    for cleaned_tweet in cleaned_tweets:
        writer.writerow([None, None, None, cleaned_tweet])
        print(cleaned_tweet)


