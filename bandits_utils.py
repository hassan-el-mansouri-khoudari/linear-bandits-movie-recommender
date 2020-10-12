import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF

def non_neg_decomp(M, d):
    """
    Renvoie le vecteur H de la décomposition NMF telle que M = WH

    inputs:
        user (m x u NumpyArray):
            Matrice de taille (nb_users x nb_films) avec les notes des utilisateurs pour chaque film
        d (int):
            Dimension de la NMF
    
    output:
        H (d x m NumpyArray):
            Résultat de la "Non-negative matrix factorization"
    """
    model = NMF(n_components=d, init='random', random_state=0)
    W = model.fit_transform(M)
    H = model.components_
    return H

def LinUCB(T, user, H, d):
    """
    Algorithm 1 from "http://rob.schapire.net/papers/www10.pdf"

    inputs:
        T (int):
            Nombre d'itérations pour l'algorithme
        user (d x m NumpyArray):
            Notes d'un utilisateur
        H (d x m NumpyArray):
            Résultat de la "Non-negative matrix factorization" issue de la fonction "non_neg_decomp", m est le nombre de films
        d (int):
            Dimension de la NMF
    
    output:
        a_t (int): Recommendation pour le user
        (Temporaire: Pour analyse) history : liste des films choisis pendant de l'algorithme
        (Temporaire: Pour analyse) regret : valeurs du regrets
    """
    

    r = user.T
    x_t = H.T

    delta = 1/(T**2)
    alpha = 1 + np.sqrt(np.log(2/delta)/2)

    nb_bras = H.shape[1]
    
    #partie initialisation du regret
    regret = np.zeros(T)
    reward_max = np.max(r)    

    ##Initialisation des variables
    A = [np.eye(d) for i in range(nb_bras)]
    b = [np.zeros((d,1)) for i in range(nb_bras)]
    theta = [np.linalg.inv(A[a])@b[a] for a in range(nb_bras)]
    p = [(theta[a].T)@x_t[a] for a in range(nb_bras)]
    #history of movies chosen
    history = np.zeros(T)
    visits = np.zeros(nb_bras)
    
    for t in range(T):
        for a in range(nb_bras):
            theta[a] = np.linalg.inv(A[a])@b[a]
            p[a] = (theta[a].T)@x_t[a] + alpha*np.sqrt(((x_t[a].T)@np.linalg.inv(A[a]))@x_t[a])
        a_t = np.argmax(p)
        A[a_t] = (A[a_t] + x_t[a_t]@(x_t[a_t].T)).reshape((d,d))
        b[a_t] = b[a_t] + r[a_t]*(x_t[a_t]).reshape((d,1))
        
        #analysis part
        visits[a_t] += 1
        history[t] = a_t
        regret[t] = (t+1)*reward_max - r@visits

    return a_t, history, regret

def UCB(i, delta, mu, T):
    """
    Renvoie la valeur de la fonction UBC_i, eq 7.2, page 102

    inputs:
        i (int):
            Numéro du bras
        delta (float):
            Valeur de delta (eq 7.1)
        mu (1 x k NumpyArray):
            Vecteur de la moyenne empirique de chacun des bras
        T (1 x k NumpyArray):
            Vecteur contenant le nombre de fois où chaque bras a été tiré
    
    output:
        float:
            Valeur de la UCB pour le bras i

    """
    if T[i] == 0:
        return np.inf
    else:
        return mu[i] + np.sqrt(2 * np.log(1/delta)/T[i])

def UCB_tot(n, delta, k, means, std):

    """
    Implémentation de l'algorithme UCB(delta), Algo 3 page 102

    inputs:
        n (int):
            Nombre d'itérations
        delta (float):
            Valeur de delta (eq 7.1)
        k (int):
            Nombre de bras
        means (1 x k NumpyArray):
            Vecteur des moyennes
        std (1 x k NumpyArray):
            Vecteur des variances
    
    output:
        X (1 x n NumpyArray):
            Réalisations (X_t)_t
        Rn (1 x n NumpyArray):
            Regrets (R_t)_t
        A (1 x n NumpyArray):
            Bras choisis
        mu (1 x k NumpyArray):
            Moyennes empiriques de chaque bras    

    """
    X = np.zeros(n) ##Réalisations (X_t)_t
    T = np.zeros(k) ##T[i] = nb de fois où i a été tiré
    arms_mat = np.zeros((n,k))
    mu = np.zeros(k) ##mu[i] = moyenne empirique de i

    A = np.zeros(n, dtype = int)
    
    ##Compute Rn at each iteration
    Rn = np.zeros(n)
    mu_star = np.max(means)
    for t in range(n):
        all_A_t = np.zeros(k)
        for i in range(k):
            all_A_t[i] = UCB(i, delta, mu, T) ##On calcul UCB_i pour tous les bras
        A[t] = np.argmax(all_A_t) ##On prend le bras qui a le UCB_i le plus haut
        X[t] = launch_arm(A[t]) ##On génère X_t la réalisation à partir du bras choisi
        T[A[t]] += 1 ##Le bras i a été tiré une fois de plus
        arms_mat[t, A[t]] = X[t] ##On stock la valeur de la réalisation dans une matrice
        mu[A[t]] = 1/T[A[t]] * sum(arms_mat[:, A[t]]) ##On update la moyenne empirique
        Rn[t] = (t+1)*mu_star - T@means ##On calcul le regret à l'instant t
    return X, Rn, A, mu

def launch_arm(i):
    """
    Lance le bras i

    input:
        i (int):
            Indice du bras

    output:
        float:
            X_t ~ N(means[i], std[i])
    """
    ## La distribution doit être 1 sub-ussian
    return np.random.normal(means[i],std[i])