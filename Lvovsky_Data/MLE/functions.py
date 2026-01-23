import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import math
import cmath
from scipy.integrate import simpson, romb
from scipy.stats import iqr, norm, gaussian_kde
from scipy.special import erfc, wofz
import pickle
import os

def P_x_theta(x, theta, alpha, r2, p = 0.6810463):

    tau = np.sqrt(1-r2)
    beta = 1
    W0 = 1 / np.sqrt(np.pi) / beta * np.exp(- x**2 / beta**2)

    
    c0 = tau / (np.sqrt(tau**2 + alpha**2))
    c1 = alpha / (np.sqrt(tau**2 + alpha**2))
    mu = r2 * np.cos(theta)
    correct_term = (1 - p * c1**2 + 2 * p * c1**2 * x**2 / beta**2 + 2 * np.sqrt(2) * p * c0 * c1 * mu * x / beta**2)

    # correct_term = 1
    Px = W0 * correct_term 

    return Px


def pdf_kernel_optimal_bandwidth_histogram(X, X_samples, kernel_method, bandwidth_method):

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

def data_wrapper(theta_list, x_list, N_theta):
    grouped_data_dict = {i: [] for i in range(N_theta)}

    theta_mod = np.mod(theta_list, np.pi)
    theta_even_odd = np.floor(theta_list / np.pi).astype(int) % 2
    bin_width = np.pi / N_theta
    bin_indices = np.floor(theta_mod / bin_width).astype(int)

    # bin_indices = np.clip(bin_indices, 0, N_theta - 1)

    for bi, x, eo in zip(bin_indices, x_list, theta_even_odd):
        grouped_data_dict[int(bi)].append(float(x) * ((-1) ** (eo+1)))

    grouped_theta_list = []
    for i in range(N_theta):
        grouped_theta_list.append((i + 0.5) * bin_width)

    return grouped_theta_list, grouped_data_dict


def reconstruct_pdf_mle(X_samples, X_precision, gmm_components):
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


def phi_1(theta, alpha, r2, radius, t=1, p=0.6810463):
    mu = radius * np.cos(theta)
    nu = radius * np.sin(theta)
    tau = np.sqrt(1-r2)
    c0 = tau / (np.sqrt(tau**2 + alpha**2))
    c1 = alpha / (np.sqrt(tau**2 + alpha**2))
    beta = np.abs(radius)
    phi0 = np.exp(- t**2 * beta**2 / 4)
    return phi0 * (1 - p * c1**2 * beta**2 * t**2 / 2 - 1j * np.sqrt(2) * p * c0 * c1 * t * mu)


# def compute_phi_at_t(w_values, X_precision, t=1):
#     """
#     Computes the characteristic function φ(t) at t = 1 (by default) using numerical integration.
#     """
#     # dx = 2 * L / n
#     # x_values = np.linspace(-L, L - dx, n)

#     # Compute φ(t)
#     integrand = w_values * np.exp(1j * X_precision * t)
#     # phi_t1 = np.sum(integrand) * dx  # Trapezoid or midpoint integration (simple Riemann sum)
#     phi_t1 = np.trapz(integrand, x=X_precision)  # Trapezoid integration
#     return phi_t1


# def compute_gaussian_phi_at_t_optimal(mu, nu, X_precision, X_samples, w_gaussian_values, t):
#     """
#     Computes the Gaussian kernel estimated characteristic function φ(t) at t = 1 (by default) with optimized bandwidth.
#     """
    
#     # Compute φ(t)
#     phi_t1 = compute_phi_at_t(w_gaussian_values, X_precision, t)
#     # optimal_bd = 2 / (mu**2 + nu**2)  * np.sqrt( - np.log(1 - ( ( 1 - np.abs(phi_t1)**2 ) / 2 / len(X_samples) / np.abs(phi_t1) **2 )))
#     optimal_bd = 2 / (mu**2 + nu**2) / t * np.sqrt( (1 - np.abs(phi_t1) **2) / ( 2 * len(X_samples) * np.abs(phi_t1) **2) )

#     phi_K = np.exp( - optimal_bd**2 * t**2 * (mu**2 + nu**2) / 4)
#     phi_nk = 1 / len(X_samples) * np.sum(np.exp(1j * X_samples * t))
#     phi_t1_optimal = phi_nk * phi_K
#     return phi_t1_optimal, optimal_bd


def compute_t_fourier_transform(w_values, X_precision, t):
    """
    Computes the characteristic function φ(t) at given t using numerical integration.
    """

    # Compute φ(t)
    integrand = w_values * np.exp(1j * X_precision * t)
    phi_t = simpson(integrand, x=X_precision)  # Simpson's rule integration
    return phi_t


def rho_kitten(y, yp, r2, alpha, p = 0.6810463):

    tau = np.sqrt(1-r2)
    c0 = tau / (np.sqrt(tau**2 + alpha**2))
    c1 = alpha / (np.sqrt(tau**2 + alpha**2))
    term1 = np.exp(- (y**2 + yp**2) / 2)
    term2 = 1 - p * c1**2 - np.sqrt(2) * p * c0 * c1 * (y + yp) + 2 * p * c1**2 * y * yp
    return 1 / np.sqrt(np.pi) * term1 * term2


def rho_max(grouped_data_dict, mu_max, N_mu, kappa, n_precision, g_list, h_list, L_limit, N_theta, gmm_components, correction = False):


    mu_list = np.linspace(-mu_max, mu_max, N_mu)
    rho_heat_map = np.zeros((len(h_list), len(g_list)), dtype=complex)
    exp_term = np.exp(-1j * mu_list[:, np.newaxis] * (h_list / 2)[np.newaxis, :])


    for idx_g, g in enumerate(tqdm(g_list)):

        phi_t1_over_mu_list = np.zeros(N_mu, dtype=complex)
    
        for idx_mu, mu in enumerate(mu_list):
            
            phi = np.arctan(g/mu)
            theta = np.where(phi<=0, phi, phi + np.pi)      
            eps = 1e-12
            cos_th = np.cos(theta)
            sin_th = np.sin(theta)
            radius = np.where(np.abs(cos_th) > eps, mu / cos_th, g / sin_th)

            theta_bin = int(np.floor(np.mod(theta, np.pi) / (np.pi / N_theta)))
            xb = grouped_data_dict[theta_bin]
            X_samples = np.array(xb)
            

            try:

                X_precision = np.linspace(-L_limit, L_limit, n_precision)

                W_Gaussian = reconstruct_pdf_mle(X_samples, X_precision, gmm_components)

                if correction == True:
                    phi_t1_vals_gaussian = compute_t_fourier_transform(W_Gaussian, X_precision, t = radius/kappa)
                    phi_t1_vals_gaussian *= 1 / np.exp( - ((1 - kappa) / kappa * radius)**2 / 2   )
                else:
                    phi_t1_vals_gaussian = compute_t_fourier_transform(W_Gaussian, X_precision, t = radius)

            except:
                phi_t1_vals_gaussian = 1

            phi_t1_over_mu_list[idx_mu] = phi_t1_vals_gaussian

        rho_element = (1 / (2 * np.pi)) * np.trapz(phi_t1_over_mu_list[:, np.newaxis] * exp_term, mu_list, axis=0)

        
        rho_heat_map[:, idx_g] = rho_element

            
    return rho_heat_map