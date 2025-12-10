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
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize


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


def samplings_with_pdf(pdf, X, n):

    X_samples = np.random.choice(X, size=n, p=pdf / np.sum(pdf))
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


def reconstruct_pdf_mle(X_samples, X_precision, gmm_components=3):
    """
    MLE methods using Gaussian Mixture Model to reconstruct the pdf from samples.
    """
    X_samples = np.asarray(X_samples, float).ravel()
    X_grid    = np.asarray(X_precision, float).ravel()

    
    gmm = GaussianMixture(
        n_components=gmm_components
    ).fit(X_samples[:, None])

    pdf_hat = np.exp(gmm.score_samples(X_grid[:, None]))  # continuous pdf on grid
    # tiny renorm for neat comparisons
    Z = simpson(pdf_hat, X_grid)
    if np.isfinite(Z) and Z > 0:
        pdf_hat /= Z
    return pdf_hat


def compute_perfect_phi(mu, nu, alpha, t):

    """
    Computes perfect (analytical) phi.
    """

    summ = 0
    for j in range(1,4):
        for k in range(1,4):
            summ += phi_0(t + s_jk(j,k,mu,nu,alpha), mu, nu) * np.exp(d_jk(j,k,mu,nu,alpha))
    return Nc_2(alpha) * summ


def compute_phi_at_t(w_values, X_precision, t):
    """
    Computes the characteristic function φ(t) at t = 1 (by default) using numerical integration.
    """
    # dx = 2 * L / n
    # x_values = np.linspace(-L, L - dx, n)

    # Compute φ(t)
    integrand = w_values * np.exp(1j * X_precision * t)
    # phi_t1 = np.sum(integrand) * dx  # Trapezoid or midpoint integration (simple Riemann sum)
    phi_t1 = np.trapz(integrand, x=X_precision)  # Trapezoid integration
    return phi_t1


def compute_gaussian_phi_at_t_optimal(mu, nu, X_precision, X_samples, w_gaussian_values, t):
    """
    Computes the Gaussian kernel estimated characteristic function φ(t) at t = 1 (by default) with optimized bandwidth.
    """

    # Compute φ(t)
    phi_t1 = compute_phi_at_t(w_gaussian_values, X_precision, t)
    # optimal_bd = 2 / (mu**2 + nu**2)  * np.sqrt( - np.log(1 - ( ( 1 - np.abs(phi_t1)**2 ) / 2 / len(X_samples) / np.abs(phi_t1) **2 )))
    optimal_bd = 2 / (mu**2 + nu**2) / t * np.sqrt( (1 - np.abs(phi_t1) **2) / ( 2 * len(X_samples) * np.abs(phi_t1) **2) )

    phi_K = np.exp( - optimal_bd**2 * t**2 * (mu**2 + nu**2) / 4)
    phi_nk = 1 / len(X_samples) * np.sum(np.exp(1j * X_samples * t))
    phi_t1_optimal = phi_nk * phi_K
    return phi_t1_optimal, optimal_bd



