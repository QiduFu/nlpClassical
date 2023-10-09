#Logistic regression with NLP
#By: Qidu (Quentin) Fu
#Note: This is based on the assignment 1 of the coursera 
#NLP specialization's week1 one assignment


#Import the necessary libraries ------------------------------------------------
#-------------------------------------------------------------------------------
import nltk
import pandas as pd
import numpy as np

nltk.download('twitter_samples')
from nltk.corpus import twitter_samples

nltk.download('stopwords')
from nltk.corpus import stopwords

from utils import process_tweet, build_freqs

#Prepare the data --------------------------------------------------------------
#-------------------------------------------------------------------------------

#select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#split the data into two pieces, one for training and one for testing
test_pos = all_positive_tweets[:4000]
train_pos = all_positive_tweets[4000:]
test_neg = all_negative_tweets[:4000]
train_neg = all_negative_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

#Create the numpy array of positive labels and negative labels
train_y = np.append(np.ones((len(train_pos),1)), np.zeros((len(train_neg),1)), axis = 0)
print(train_y.shape)
test_y = np.append(np.ones((len(test_pos),1)), np.zeros((len(test_neg),1)), axis = 0)
print(test_y.shape)

print("Train_y.shape ", train_y.shape)
print("Test_y.shape ", test_y.shape)

#Create frequency dictionary
freqs = build_freqs(train_x, train_y)
#Process tweet
process_tweet(train_x[0])

#Logistic regression with a NLP ------------------------------------------------
#-------------------------------------------------------------------------------
#Sigmoid function
def sigmoid(z):
    h = 1/(1+np.exp(-z))
    return h

#Gradient descent function
def gradient_descent(x, y, theta, alpha, num_iters):
    #Create a docstring
    """
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output: 
        J: the final cost
        theta: your final weight vector
        """
        
    #Get m, the number of rows in matrix x -- number of training examples
    m = x.shape[0]
    for _ in range(num_iters):
        #Get the dot product of x and theta
        z = np.dot(x, theta)
        #Get the sigmoid of z
        h = sigmoid(z)
        #Calculate the cost function
        J = (-1/m)*(y.T @ np.log(h) + (1-y).T@np.log(1-h))
        #Derivative of cost function with respect to theta
        dj = (1/m)*(x.T@(h-y))
        #Update theta
        theta = theta - alpha*dj
    
    J = float(J)
    return J, theta

#Extract features function
def exrtact_features(tweet, freqs, process_tweet=process_tweet):
    #Create a docstring
    """
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        process_tweet: a function to process tweet
    Output:
        x: a feature vector of dimension (1,3)
    """
    #process_tweet tokenizes, stems, and removes stopwords
    word_list = process_tweet(tweet)
    #3 elements for [bias, positive_sum, negative_sum]
    x = np.zeros(3)
    #bias terms is set to 1
    x[0] = 1
    
    for word in word_list:
        #Increment the word count for the positive label 1
        x[1] += freqs.get((word, 1.0), 0)
        #Increment the word count for the negative label 0
        x[2] += freqs.get((word, 0.0), 0)
        assert(x.shape == (3,1))
        
    x = x.reshape((1,3)) #add batch dimension for further processing
    return x


#Train the model ---------------------------------------------------------------
#-------------------------------------------------------------------------------
#Collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x),3))
for i in range(len(train_x)):
    X[i, :]= exrtact_features(train_x[i], freqs)
    
#training labels corresponding to X
Y = train_y

#Apply gradient descent
J, theta = gradient_descent(X, Y, np.zeros((3,1)), 1e-9, 1500)
print("The cost after training is ", J)
print("The resulting vector of weights is ", [theta[0][0], theta[1][0], theta[2][0]])

#Test the logistic regression ---------------------------------------------------
#-------------------------------------------------------------------------------
def predict_tweet(tweet, freqs, theta):
    #Create a docstring
    """
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    """
    #Extract the features of the tweet and store it into x
    x = exrtact_features(tweet, freqs)
    #Make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred

#Test the model on a positive tweet
def test_logistic_regression(test_x, test_y, freqs, theta):
    #Create a docstring
    """
    Input:
        test_x: a list of tweets
        test_y: (m,1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each tuple (word, label)
        theta: weight vector of dimension (3,1)
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    #The list for storing predictions
    y_hat = []
    for tweet in test_x:
        #Get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
            
    #With the above list, calculate the accuracy
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)

#Some error analysis -----------------------------------------------------------
#-------------------------------------------------------------------------------
# Some error analysis done for you
print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))

