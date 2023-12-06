import numpy as np


def process_pixels(image, lambda, nuisance):

    (Tp, Tb) = nuisance
    atm_tr = transmittance(lambda)
    emis_b = emissivity(image, lambda)
    Bp, Bb = blackbody(lambda, Tp), blackbody(lambda, Tb)

    L_tec = Bp - (emis_b * Bb)

    mask = plume_mask(image)
    differenced_img = diff(image, mask)

    processed = - np.log(1 + (differenced_img / (atm_tr * L_tec)))

    return processed

def blackbody(lambda, T):
    h = 6.62607015e-34  # Planck's constant in J·s
    c = 3.00e8         # Speed of light in m/s
    k = 1.380649e-23   # Boltzmann constant in J/K

    return (2 * h * c**2 / lambda**5) / (np.exp(h * c / (lambda * k * T)) - 1)

def transmittance(lambda):

    return atm_tr

def emissivity(image, lambda):

    return