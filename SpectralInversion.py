import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from pathlib import Path
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import scipy

import random

def spectral_inversion(
    reference_spectra,
    residual_spectra,
    lambda_,
    dataset,
    compound_list,
    post_cov = True,
    do_spilu = True,
    ):
    """
    Temporally regularised inversion using selected bassis functions.

    Args:
        reference_spectra (np.ndarray): The absorption spectra, shape (Ns, Nl)
        residual_spectra (np.ndarray): The residuals of the transmiatted spectra,
            shape (Np, Nl)
        lambda_ (float): The amount of regularisation. 0.005 seems to work?
        post_cov (boolean, optional): Return inverse posterior covariance matrix.
            Defaults to True.
        do_spilu (boolean, optional): Solve the system using and ILU factorisation.
            Seems faster and more memory efficient, with an error around 0.5-1%

    Returns:
        Maximum a poseteriori estimate, and variance. Optionally, also
        the posterior inverse covariance matrix.
    """

    Ns, Nl = reference_spectra.shape
    Np = residual_spectra.shape[0]
    assert Ns != residual_spectra.shape[1], "Wrong spectral sampling!!"
    # Create the "hat" matrix
    A_mat = build_A_matrix(reference_spectra, Ns, Nl, Np)
    # Regulariser
    D_mat = sp.lil_matrix(sp.kron(sp.eye(Ns)))
    # Squeeze observations
    y = residual_spectra.flatten()
    # Solve
    C = sp.csc_array(A_mat.T @ A_mat)

    c_inv = np.linalg.inv(C.toarray())
    sigma = c_inv.diagonal()

    Ns, Nl = reference_spectra.shape[0], reference_spectra.shape[1]

    S = []
    for i in range(Ns):
        a = sp.lil_matrix((Nl * 1, 1), dtype=np.float32)
        for j in range(1):
            a[(j * Nl) : (j + 1) * Nl, j] = reference_spectra[i, :]
        S.append(a)

    A_mat_single = sp.hstack(S)
    C_single = sp.csc_array(A_mat_single.T @ A_mat_single)
    ata = A_mat_single.T @ A_mat_single
    ata = ata.toarray()
    U, s, VT = np.linalg.svd(ata)
    np.save('/home/luke/data/Model/results/'+dataset+'/SVD.npy', {'U': U, 's': s, 'VT': VT})

    c_inv = np.linalg.inv(C_single.toarray())
    std_devs = np.sqrt(np.diag(c_inv))

    # Use broadcasting to calculate the correlation matrix
    inv_corr = c_inv / np.outer(std_devs, std_devs)

    plt.imshow(inv_corr, cmap = 'seismic', vmin=-1, vmax=1)
    ticks = np.linspace(-0.5 + inv_corr.shape[0]/(2*len(compound_list)), -0.5 + inv_corr.shape[0] * (1 - (1/(2*len(compound_list)))), len(compound_list))
    plt.xticks(ticks, compound_list, rotation=45)
    plt.yticks(ticks, compound_list)

    cbar = plt.colorbar(pad=0.01)
    cbar.set_label('Correlation Index', rotation = -90, labelpad=12)

    plt.tight_layout()

    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/Correlation_Matrix.png')
    plt.savefig('/home/luke/data/Model/plots/'+ dataset + '/Correlation_Matrix.pdf')
    plt.close()

    cobj = spl.spilu(C)
    x_sol = cobj.solve(A_mat.T @ y) if do_spilu else spl.spsolve(C, A_mat.T @ y)

    return (x_sol, sigma, C) if post_cov else (x_sol, sigma)

def build_A_matrix(spectra, Ns, Nl, Np):
    """Builds the A matrix from the spectra.

    Args:
        spectra (np.ndarray): Array of reference spectra ("templates")
        Ns (int): Number of species
        Nl (int): Number of wavelength
        Np (int): Number of pixels

    Returns:
        A sparse Nl*Nt, Ns*Nt matrix
    """
    S = []
    for i in range(Ns):
        a = sp.lil_matrix((Nl * Np, Np), dtype=np.float32)
        for j in range(Np):
            a[(j * Nl) : (j + 1) * Nl, j] = spectra[i, :]
        S.append(a)
    return sp.hstack(S)  # (Nl*Np, Ns*Np) matrix