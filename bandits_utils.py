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