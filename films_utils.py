import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

def fill_matrix_MF(R, d, max_iter = 1000):
    """
    Returns a predicted matrix using matrix factorization to deal with zeros(non rated movies)
    """

    model = NMF(n_components=d, init='random', random_state=0, max_iter = max_iter)

    U = model.fit_transform(R)
    M = model.components_

    return U@M

def get_labels_from_clustering(nb_clusters, R):
    """
    Applies K-means and returns the labels
    """
    return KMeans(nb_clusters).fit(R).labels_

def get_best_movies_by_cluster(R, labels, nb_clusters):
    clusters_rating_mean = R.T@np.eye(nb_clusters)[labels]
    for i in range(nb_clusters):
        clusters_rating_mean[:,i] = clusters_rating_mean[:,i]/np.count_nonzero(labels == i)
    return np.argsort(-1*clusters_rating_mean, axis=0)

def get_reward_film(movie, user):
    return user[movie]

def UCB_film(cluster_index, delta, mu, T):
    if T[cluster_index] == 0:
        return np.inf
    else:
        return mu[cluster_index] + np.sqrt(2 * np.log(1/delta)/T[cluster_index])

def UCB_tot_film(n, new_user, delta, best_movies_by_cluster, k):
    X = np.zeros(n) ##Réalisations (X_t)_t
    T = np.zeros(k, dtype = int) ##T[i] = nb de fois où i a été tiré
    arms_mat = np.zeros((n,k))
    mu = np.zeros(k) ##mu[i] = moyenne empirique de i

    A = np.zeros(n, dtype = int)
    
    for t in range(n):
        all_A_t = np.zeros(k)

        ##On calcul UCB_film pour tous les clusters de films
        for i in range(k):
            all_A_t[i] = UCB_film(i, delta, mu, T) 

        ##On prend le bras qui a le UCB_i le plus haut
        A[t] = np.argmax(all_A_t)

        ##On génère X_t la réalisation à partir du bras choisi
        movie = best_movies_by_cluster[:,A[t]][T[A[t]]]
        X[t] = get_reward_film(movie,new_user)

        ##Le bras i a été tiré une fois de plus
        T[A[t]] += 1 
        arms_mat[t, A[t]] = X[t] ##On stock la valeur de la réalisation dans une matrice
        mu[A[t]] = 1/T[A[t]] * sum(arms_mat[:, A[t]]) ##On update la moyenne empirique
    return X, A, T, mu