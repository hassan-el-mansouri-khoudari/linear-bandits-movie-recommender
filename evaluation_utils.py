import numpy as np
from films_utils import UCB_tot_film
import random


def normalized_DGC(rewards_user, user):
    """
    return GDC user /IGDC for the NDGC gain
    """
    sorted_real = -1*np.sort(-1*user)
    IGDC_user = sorted_real[0] + np.sum(sorted_real[1:]/np.log(range(2,len(sorted_real)+1)))
    GDC_user = rewards_user[0] + np.sum(rewards_user[1:]/np.log(range(2,len(rewards_user)+1)))
    return GDC_user/IGDC_user




def gain_NDGC_UCB(n, R, delta, best_movies_by_cluster, nb_clusters):
    """
    Return the normalized GDC over all users
    """
    
    sum = 0
    N = R.shape[0]
    for i in range(N):
        rewards_user, _, _, _= UCB_tot_film(n, R[i], delta, best_movies_by_cluster, nb_clusters)
        sum += normalized_DGC(rewards_user,R[i])
    mean = sum/N
    return mean


def best_average_rewards(R,user,nb_rounds):
    """
    returns the rewards for a user of the best films (according to the mean of the ratings matrix)
    """
    rewards_user = []
    arg_mean_ratings = np.argsort(-1*np.mean(R,axis=0))
    for i in range(nb_rounds):
        rewards_user.append(R[user][arg_mean_ratings[i]])
    return rewards_user


def random_recommandation_rewards(R, user,nb_rounds):
    """
    returns the rewards for a user according to a random selection of movies
    """
    rewards_user = []
    recommended = [i for i in range(R.shape[1])]
    for i in range(nb_rounds):
        m = random.choice(recommended)
        recommended.remove(m)
        rewards_user.append(R[user][m])
    return rewards_user


def gain_NDGC(R, delta, best_movies_by_cluster, nb_clusters, nb_rounds, method ):
    """
    Calculates the gain NDGC given a method of recomandatation
    """
    sum = 0
    N = R.shape[0]
    for i in range(N):
        if method == "random":
            rewards_user = random_recommandation_rewards(R, i, nb_rounds=nb_rounds)
        elif method == "UCB":
            rewards_user, _, _, _, _= UCB_tot_film(nb_rounds, R[i], delta, best_movies_by_cluster, nb_clusters)
        elif method == "best_average":
            rewards_user = best_average_rewards(R, i, nb_rounds=nb_rounds)

            
        sum += normalized_DGC(rewards_user,R[i])
    mean = sum/N
    return mean


