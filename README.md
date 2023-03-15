# Sentiment analysis of climate change tweets

## Clean and preprocess tweet CSV files
Before performing any analysis on the tweet data, the tweets in the CSV file must be cleaned and preprocessed.
1. Choose a CSV file from the tweets directory and run step1_prepro.py on your chosen tweet CSV file.
1. Run step2_prepro.py on the output CSV file of step1_prepro.py.

## Word counts
- count_vectorizer.py gets the top 10 most common words across all tweets given a cleaned data set.
- TF-IDF.py gets the top 10 most significant words across all tweets given a cleaned data set.

## Lexicon-based analysis methods
Run the appropriate script on your cleaned and preprocessed data set based on the lexicon-based method that you want to use to output a sentiment analysis graph.
- TEXTBLOB.py
- SENTIWORD.py
- VADER.py

## Learning-based analysis methods
1. Run label_data.py on your cleaned and preprocessed data, then adds a column with a sentiment label based on the lexicon-based VADER method.
1. Choose either the Naive Bayes method or SVM method and run the appropriate .py script.
- NAIVE-BAYES.py
- SVM.py
  1. Given a training data set, builds a model and outputs an accuracy assessment and a confusion matrix by comparing it to the sentiment label column.
  1. Given an input data set, performs sentiment analysis based on the previously generated model.