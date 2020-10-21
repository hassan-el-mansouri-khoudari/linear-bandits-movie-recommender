"""
Ce fichier sert a load le dataset des movies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bandits_utils import *

def f_load_dataset():
    """
    Load le dataset movie dans la matrice Ratings
    """
    ratings = pd.read_table('ml-1m/ratings.dat', sep='::', 
                            names = ['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding = 'latin1',
                            engine = 'python')
    movies  = pd.read_table('ml-1m/movies.dat',  sep='::',
                            names = ['MovieID', 'Title', 'Genres'], 
                            encoding = 'latin1',
                            engine ='python')
    users   = pd.read_table('ml-1m/users.dat',  sep='::', 
                            names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip'], 
                            encoding = 'latin1',
                            engine = 'python')

    N = 1000
    ratings_count = ratings.groupby(by='MovieID', as_index=True).size()
    # top_ratings = ratings_count.sort_values(ascending=False)[:N]
    top_ratings = ratings_count[ratings_count>=N]

    movies_topN = movies[movies.MovieID.isin(top_ratings.index)]
    ratings_topN = ratings[ratings.MovieID.isin(top_ratings.index)]

    n_users = ratings_topN.UserID.unique().shape[0]
    n_movies = ratings_topN.MovieID.unique().shape[0]

    R_df = ratings_topN.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)

    Ratings = np.array(R_df) ##M = R_df.as_matrix()
    return Ratings