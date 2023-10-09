#This is based on the assignment 1 of 
#the coursera NLP specialization's week1 one assignment

import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Clean the text, tokenize it into separate words,
# remove stopwords, and convert words to stems


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    # remove stock market tickers like $GE with empty string
    tweet = re.sub(r"\$\\w*", "", tweet)
    # remove old style retweet text 'RT' with empty string
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # remove hyperlinks
    tweet = re.sub(r"https?//[^\s\n\r]", "", tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r"#", "", tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (
            word not in stopwords_english
            and word not in string.punctuation  # remove stopwords
        ):  # remove punctuation
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

#This counts how often a word in the corpus
#was associated with a positive label (1) or a negative label (0)
#It builds the freqs dictionary, where each key is a (word, label) tuple,
#and the value is the count of its frequency within the corpus of tweets.
def build_freqs(tweets, ys):
    """Build frequencies.

    Args:
        tweets (list): a list of tweets
        ys (ndarray): an mX1 array with the sentiment label of each tweet

    Returns:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    """
    #Convert np array to list since zip needs an iterable.
    #The squeeze is necessary or the list ends up with one element.
    #Also note that this is just a NOP if ys is already a list
    yslist = np.squeeze(ys).tolist()
    
    #Start with an empty dictionary and populate it by looping over all tweets
    #and over all processed words in each tweet
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs.get(pair, 0) + 1
    return freqs

