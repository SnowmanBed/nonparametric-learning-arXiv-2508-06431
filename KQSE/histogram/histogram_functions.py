import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import math
import cmath
from scipy.integrate import simpson, romb
from scipy.stats import iqr, norm, gaussian_kde
from scipy.special import erfc, wofz
import pickle
import os

def W_0(mu, nu, X):
    return 1 / np.sqrt(np.pi * (mu**2 + nu**2)) * np.exp(-X**2 / (mu**2 + nu**2))

def phi_0(t, mu, nu):
    return np.exp(- t**2 * (mu**2 + nu**2) / 4)

def s_jk(j, k, mu, nu, alpha):
    return np.sqrt(2) * ( (nu - 1j * mu) * np.conjugate(alpha) * np.exp(-2*np.pi*1j/3 * (k-1)) - (nu + 1j * mu) * alpha * np.exp(2*np.pi*1j/3 * (j-1))) / (mu**2 + nu**2)

def d_jk(j, k, mu, nu, alpha):
    return - np.abs(alpha) ** 2 + ( (nu + 1j * mu)**2 * alpha ** 2 * np.exp(4*np.pi*1j/3*(j-1)) + (nu - 1j * mu)**2 * np.conjugate(alpha)**2 * np.exp(-4*np.pi*1j/3*(k-1)) ) / 2 / (mu**2 + nu**2)

def Nc_2(alpha):
    Nc_2_inv = 0
    for j in range(1,4):
        for k in range(1,4):
            Nc_2_inv += np.exp( - np.abs(alpha)**2 + np.abs(alpha)**2 * np.exp(2 * np.pi * 1j * (k - j) / 3))
    return 1 / Nc_2_inv  


def normalized_cat_state_pdf(mu, nu, alpha, X):

    summ = 0
    for j in range(1,4):
        for k in range(1,4):
            summ += np.exp( 1j * X * s_jk(j,k, mu, nu, alpha) ) * np.exp( d_jk(j,k, mu, nu, alpha) )
    return np.real( W_0(mu, nu, X) * Nc_2(alpha)* summ )

def samplings_with_pdf(mu, nu, alpha, X, n):

    pdf = normalized_cat_state_pdf(mu, nu, alpha, X)
    X_samples = np.random.choice(X, size=n, p=pdf/np.sum(pdf))
    return X_samples

def samplings_with_noisy_pdf(pdf, X, n, kappa):

    X_samples = np.random.choice(X, size=n, p=pdf / np.sum(pdf))
    X_samples = kappa * X_samples + np.random.normal(0, 1, n) * (1 - kappa)  # Add noise
    return X_samples

def find_suitable_L(mu, nu, alpha):


    L_trial = 60
    X = np.linspace(0, L_trial, 100)
    
    pdf_trial = normalized_cat_state_pdf(mu, nu, alpha, X)

    indices = np.where(pdf_trial < 1E-7)[0]
    if indices.size == 0:
        L = L_trial
    else:
        L = X[indices[0]]
    

    return L


def pdf_kernel_optimal_bandwidth(X, X_samples, kernel_method, bandwidth_method):

    """
    kernel_method: 'gaussian' or 'epanechnikov'
    bandwidth_method: 'cv' for cross-validation or 'silverman' for Silverman's rule of thumb
    """

    if bandwidth_method == 'cv':
        fold = 5
        bw_precisions = 30
        bandwidths = np.logspace(-1, 1, bw_precisions)  # Testing range of bandwidths
        grid = GridSearchCV(KernelDensity(kernel=kernel_method), {'bandwidth': bandwidths}, cv=fold)  # 5-fold cross-validation
        grid.fit(X_samples[:, None])
        bd = grid.best_params_['bandwidth']
    
    elif bandwidth_method == 'silverman':
        silverman_bandwidth = 0.91 * min(np.std(X_samples, ddof = 1), iqr(X_samples) / 1.34) * len(X_samples) ** (-1 / 5)
        bd = silverman_bandwidth

    kde = KernelDensity(kernel=kernel_method, bandwidth=bd).fit(X_samples[:, None])
    w_values = np.exp(kde.score_samples(X.reshape(-1,1)))

    return w_values, bd