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

def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n

def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        en = my_file.loc[i][0]
        fr = my_file.loc[i][1]
        etof[en] = fr

    return etof


def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos

# Procedure to plot and arrows that represents vectors with pyplot
def plot_vectors(vectors, colors=['k', 'b', 'r', 'm', 'c'], axes=None, fname='image.svg', ax=None):
    scale = 1
    scale_units = 'x'
    x_dir = []
    y_dir = []
    
    for i, vec in enumerate(vectors):
        if vec.shape[0] == 1:     # row vector
            x_dir.append(vec[0][0])
            y_dir.append(vec[0][1])
        elif vec.shape[1] == 1:   # column vector
            x_dir.append(vec[0][0])
            y_dir.append(vec[1][0])
    
    if ax == None:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax
      
    if axes == None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]
        
    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])
        
    for i, vec in enumerate(vectors):
        if vec.shape[0] == 1:     # row vector
            ax2.arrow(0, 0, vec[0][0], vec[0][1], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
        elif vec.shape[1] == 1:   # column vector
            ax2.arrow(0, 0, vec[0][0], vec[1][0], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
    
    if ax == None:
        plt.show()
        fig.savefig(fname)
