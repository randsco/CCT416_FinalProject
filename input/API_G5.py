import tweepy
import csv
import pandas as pd
import json

# Authenticate with the Twitter API using Consumer Key and Consumer Secret
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""


# Authenticate with the Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)


# Set the topic you want to search for
search_topic = "#climatechange OR climate change OR climate crisis OR global warming lang:en exclude:replies exclude:retweets"

# Initialize variables to keep track of the tweets retrieved
all_tweets = []
tweets = api.search_tweets(q=search_topic, count=100, tweet_mode='extended')
all_tweets.extend(tweets)

# Keep paginating through the tweets until 1000 tweets are retrieved
while len(all_tweets) < 1000:
    max_id = all_tweets[-1].id - 1
    tweets = api.search_tweets(q=search_topic, count=100, max_id=max_id, tweet_mode='extended')
    all_tweets.extend(tweets)


# Create a dataframe
columns = ['Time', 'User', 'Tweet', 'Location', 'Retweets']
data = []
for tweet in all_tweets:
    data.append([
        tweet.created_at,
        tweet.user.screen_name,
        tweet.full_text,
        tweet.user.location,
        tweet.retweet_count])

df = pd.DataFrame(data, columns=columns)

df.to_csv('G5-2.csv', encoding='utf-8')
