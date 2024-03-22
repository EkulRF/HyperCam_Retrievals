import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl

import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

from radis import calc_spectrum
from radis import load_spec
from radis.test.utils import getTestFile
from radis import Spectrum
from radis.tools import convolve_with_slit

def getReferenceMatrix(Compounds: dict, T: float, P: float, W_obs: np.ndarray, sigma: float, dataset: str) -> np.ndarray:
    """
    Generate a reference matrix based on input compounds and parameters.

    Args:
        Compounds (dict): A dictionary containing information about chemical species.
        T (float): Temperature in Kelvin.
        P (float): Pressure in bar.
        W_obs (np.ndarray): The wavenumber array for observed spectra.
        sigma (float): ILS width.

    Returns:
        np.ndarray: A reference matrix containing spectra of the specified compounds.

    This function generates a reference matrix by simulating and processing spectra for
    each compound defined in the 'Compounds' dictionary. It applies broadening, via a convolution with a
    Gaussian, defined by the 'sigma' parameter, and the resulting spectra are stored in the reference matrix.

    """
    output = []

    for c in Compounds:

        plt.figure()

        tmp = np.zeros_like(W_obs)

        for i in range(len(Compounds[c]['bounds'])):
            bound = Compounds[c]['bounds'][i]
            try:
                s = calc_spectrum(
                    bound[0], bound[1],  # cm-1
                    molecule=c,
                    pressure=P,  # bar
                    Tgas=T,  # K
                    mole_fraction=10**(-6),
                    path_length=100,  # cm
                    warnings={'AccuracyError':'ignore'},
                )
            except Exception as error:
                print("An exception occurred:", error)
                continue

            #s.apply_slit(0.241, 'cm-1', shape="gaussian")  # Simulate an experimental slit
            w, A = s.get('absorbance', wunit='cm-1')

            #### Instrument Lineshape Application #####

            # Create a Gaussian kernel based on your parameters
            # x_values = np.arange(-3, 3, 0.001)
            # gaussian_mu, gaussian_sigma = 0, 0.241
            # gaussian_kernel = gaussian(x_values, gaussian_mu, sigma)

            # # Normalize the kernel to have unit area
            # gaussian_kernel /= np.max(gaussian_kernel)

            # kernel = gaussian_kernel
            
            # w, A = convolve_with_slit(w, A, x_values, kernel, wunit='cm-1')

            s = Spectrum.from_array(w, A, 'absorbance', wunit='cm-1', unit='')

            iloc, jloc = np.argmin(np.abs(w.min() - W_obs)), np.argmin(np.abs(w.max() - W_obs))
            s.resample(W_obs[iloc:jloc], energy_threshold=2)

            w, A = s.get('absorbance', wunit='cm-1')

            tmp[iloc:jloc] = A

        output.append(tmp)

    ref_mat = np.array(output)
    return ref_mat