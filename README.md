# Classical NLP Models 
By Qidu Fu

## Outline
- [1 Introduction](#1)
- [2 Methodology](#2)
- [3 Acknowledgements](#3)

<a name='1'></a>
## 1 Introduction 
This repository includes four projects, specifically, these two project aim for the tasks below respectively:
- project1 (logRegressionForSentimentAnalysis): Implement logistic regression from scratch to predict sentiment polarity of Twitter text data
- project2 (naiveBayesForSentimentAnalysis): Implement naive Bayes from scratch to predict sentiment polarity of Twitter text data
- project3 (wordEmbeddings): Implement word2vec from scratch to predict analogies between words
- project4 (naiveMachineTranslationAndLSH): Implement machine translation from scratch using word embeddings and locality sensitive hashing

<a name='2'></a>
## 2 Project Methodology
To complete the aforementioned tasks, my project code files:
- Project 1 ([code file](logRegressionForSentimentAnalysis.py)):
    - clean the Twitter data
    - extract features from the data
    - implement logistic regression from scratch
    - predict sentiment polarity of Twitter text data
    - perform model evaluation
- Project 2 ([code file](naiveBayesForSentimentAnalysis.py)):
    - clean the Twitter data
    - extract features from the data
    - implement naive Bayes from scratch
    - predict sentiment polarity of Twitter text data
    - perform model evaluation
- Project 3 ([code file](wordEmbeddings.py)):
    - predict analogies between words
    - use pca to reduce the dimensionality of the word embeddings and plot them in two dimensions
    - compare word embeddings by using a similarity measure (cosine similarity)
    - understand how these vector space models work
    - perform model evaluation
- Project 4 ([code file](naiveMachineTranslation.py)):
    - generate word embeddings and transform matrices
    - perform machine translation as linear transformation of word embeddings
    - perform model evaluation

Note that the some helper functions are written in the [`utils.py`](utils.py) file. Please refer to the code and utils files for more details.

<a name='3'></a>
## 3 Acknowledgements
These are assignments completed for Coursera's NLP specialization offered by deeplearning.ai. The code base, dataset, and problem statements are from the course. The code is written by myself.
