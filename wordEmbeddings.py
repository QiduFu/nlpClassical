# Words Embeddings with NLP
# By: Qidu (Quentin) Fu
# Note: This is based on the assignment 3 of the coursera
# NLP specialization's week 3 assignment

# Import the necessary libraries ------------------------------------------------
# -------------------------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("capitals.txt", delimiter=" ")
data.columns = ["city1", "country1", "city2", "country2"]
data = data.dropna()
data = data.reset_index(drop=True)
data.head(5)

word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))


# Predict relationships among words ------------------------------------------------
# -------------------------------------------------------------------------------
def cosine_similarity(A, B):
    # Create docstring
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    """
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)
    return cos


def euclidean(A, B):
    # Create docstring
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        euclidean: numerical number representing the euclidean distance between A and B.
    """
    euclidean = np.linalg.norm(A - B)
    return euclidean


def get_country(
    city1, country1, city2, country2, embeddings, cosine_similarity=cosine_similarity
):
    """
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their emmbeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """
    group = set((city1, country1, city2))
    # get the embedding for each word in the group
    city1_emb = embeddings[city1]
    # get embedding for country1
    country1_emb = embeddings[country1]
    # get embedding for city2
    city2_emb = embeddings[city2]
    # get embedding for country2
    vec = country1_emb - city1_emb + city2_emb

    # Initialize the maximum similarity to a negative number
    max_sim = -1000

    country = ""
    for word in embeddings.keys():
        if word not in group:
            word_emb = embeddings[word]
            sim = cosine_similarity(vec, word_emb)
            if sim > max_sim:
                max_sim = sim
                country = (word, max_sim)

    return country


# Test your function cosine_similarity() -----------------------------------------
# -------------------------------------------------------------------------------
def get_accuracy(word_embeddings, data, get_country=get_country):
    """
    Input:
        word_embeddings: a dictionary where the keys are words and values are their emmbeddings
        data: a pandas dataframe containing the country and city columns
        get_country: a function that takes in 4 parameters (city1, country1, city2, country2)
    Output:
        accuracy: the accuracy of the model
    """

    num_correct = 0
    # loop through each row of the data
    for _, row in data.iterrows():
        city1 = row["city1"]
        country1 = row["country1"]
        city2 = row["city2"]
        country2 = row["country2"]
        # get the predicted country
        predicted_country, _ = get_country(
            city1, country1, city2, country2, word_embeddings
        )
        # if the prediction is correct, increase the number of correct predictions
        if predicted_country == country2:
            num_correct += 1

    # get the accuracy by dividing the number of correct predictions by the number of predictions
    accuracy = num_correct / len(data)
    return accuracy


# Plot the vectors using PCA ------------------------------------------------------
# -------------------------------------------------------------------------------
def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # mean center the data
    X_demeaned = X - np.mean(X, axis=0)
    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO="L")
    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]
    # sort the eigen values by index
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
    # sort eigenvectors using the index
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:, 0:n_components]
    # transform the data by multiplying the transpose of the eigenvectors
    # with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(
        eigen_vecs_subset.transpose(), X_demeaned.transpose()
    ).transpose()
    return X_reduced


def plot_pca(word_embeddings, words):
    reduced_embeddings = compute_pca(word_embeddings)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.3)
    for idx, word in enumerate(words):
        plt.annotate(
            word, xy=(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]), alpha=0.3
        )
    plt.show()
