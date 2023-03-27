from general_functions import *
import numpy as np

N, S, M, mu = 2, 5, 5, 0.1
eigen_vals_vecs(initiate_phi(N, S, M), N, S, M, mu, show_evecs=True, colormap='magma', savefigs=True, unphysical=False)