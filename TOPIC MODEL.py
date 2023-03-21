import pandas as pd
import numpy as np
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load tweets from CSV file
tweets_df = pd.read_csv("output/G5-2_cleaned.csv")

# Create dictionary and corpus
dictionary = corpora.Dictionary([text.split() for text in tweets_df["Cleaned_tweet"]])
corpus = [dictionary.doc2bow(text.split()) for text in tweets_df["Cleaned_tweet"]]

# Train LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=10)

# Print out topics
for i, topic in lda_model.show_topics(num_topics=10, num_words=10, formatted=False):
    print("Topic {}: {}".format(i, " ".join([w[0] for w in topic])))

# Save topics to file
with open("topics.txt", "w") as f:
    for i, topic in lda_model.show_topics(num_topics=10, num_words=10, formatted=False):
        f.write("Topic {}: {}\n".format(i, " ".join([w[0] for w in topic])))
