import numpy as np
import sys
import matplotlib.pyplot as plt
from numba import jit, njit
from numba.types import float64, int64

def initiate_values():
    global N, S, M, lambd
    if len(sys.argv) != 5:
        N, S, M = 4, 3, 6
        lambd = 0.01 # small-but-not-too-small
    else:
        N = int(sys.argv[1])
        S = int(sys.argv[2])
        M = int(sys.argv[3])
        lambd = float(sys.argv[4])

    return N, S, M, lambd

@njit
def rand_choice_numba(prob, arr=None):
    idx = np.searchsorted(np.cumsum(prob), np.random.random(), side='right')
    if arr is not None: return arr[idx]
    else: return idx # returns index, close enough

@njit
def choose_agents(N, method='random'):

    if method == 'random': 
        speaker = np.random.randint(N)
        listener = np.random.randint(N)
        while listener == speaker: listener = np.random.randint(N)

    elif method == 'NN':
        speaker = np.random.randint(N)
        arr = np.array([speaker-2, speaker-1, speaker+1, speaker+2])
        prob = np.array([0.1, 0.4, 0.4, 0.1])
        listener = rand_choice_numba(arr=arr, prob=prob)
        listener = listener%N
        
    return speaker, listener

@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def round_numba(x, decimals, out):
    return np.round_(x, decimals, out)

@njit
def get_feedback(lambd, speaker, listener, intend, infer, method=None, social_ladder=None):

    feedback = lambd if infer == intend else - lambd

    # EXPERIMENTAL ZONE

    if method == 'status':

        judgement_factors = np.linspace(2, -1, N)
        social_index = social_ladder.index(listener)
        feedback *= judgement_factors[social_index] 
        # increase or decrease the valuation of the listener's judgement based on their place on the social ladder

    return feedback

@njit
def change_social_ladder_numba(ladder, method=None):

    if method == 'swap':
        i1, i2 = np.random.randint(len(ladder)), np.random.randint(len(ladder))
        while i2 == i1: i2 = np.random.randint(len(ladder))
        ladder[i1], ladder[i2] = ladder[i2], ladder[i1]

    elif method=='rotate_left':
        ladder = list(np.roll(ladder,-1)) # former social leader becomes outcast

    elif method=='rotate_right':
        ladder = list(np.roll(ladder, 1)) # former social outcast becomes leader

    return ladder

@njit
def calc_A(phi, N, S, M):

    psi = np.divide(phi, np.expand_dims(phi.sum(axis=2), -1))
    A = 0
    for n in range(1, N):
        A += 1/(N * (N-1) * M) * (phi * np.roll(psi, n, axis=0)).sum()
    
    return A

@njit
def calc_A_naive(phi, N, S, M):

    A = 0
    for i in range(N):

        for j in range(N):
            if j == i: continue

            psi_j = np.divide(phi[j,:,:], np.expand_dims(phi[j,:,:].sum(axis=1), -1)) # [:, np.newaxis]

            for m in range(M):
                for s in range(S):

                    A += 1/(N * (N-1) * M) * phi[i, s, m] * psi_j[s, m]

    return A

@njit
def get_Alists_numba(N, S, M, lambd, n_runs=500, thresh_frac=0.98, max_steps=1e6, period=10000):

    saved_timesteps = np.array([0.2, 0.4, 0.6, 0.8, 1.0]) * (max_steps/1e6) # [2e5, 4e5, 6e5, 8e5, 1e6]

    A_megalist, timestep_megalist, phi_list = [], [], []

    for run in range(n_runs):

        saved_phis = []

        phi = np.ones((N, S, M)) / S # phi = 1/S for all speakers
        A_list, timesteps = [], []
        social_ladder = list(range(N))

        for i in range(max_steps+1):

            intend = np.random.randint(M) # meaning intended -- rho = 1/M, uniform distribution
            speaker, listener = choose_agents(N, method='random')

            produce = rand_choice_numba(prob = phi[speaker,:,intend])
            infer = rand_choice_numba(prob = phi[listener,produce,:] / np.sum(phi[listener,produce,:]))

            # updating phi
            feedback = get_feedback(lambd, speaker, listener, intend, infer, method='status', social_ladder=social_ladder)
            phi[speaker,produce,intend] += (feedback * phi[speaker,produce,intend] * (1 - phi[speaker,produce,intend]))
            phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend]) # normalise along signal axis

            if i % 5000 == 0 and i > 0:

                A = calc_A_naive(phi, N, S, M)
                A_list.append(A); timesteps.append(i)

                # if A >= thresh_frac * (S / M): break # equilibrium
            
            if i % period == 0 and i > 0:
                social_ladder = change_social_ladder_numba(social_ladder, 'rotate_left')

            if i in saved_timesteps:
                saved_phis.append(list(phi.flatten()))

        # phi_final = list(phi.flatten())

        print(f'run {run+1}')

        A_megalist.append(A_list); timestep_megalist.append(timesteps); phi_list.append(saved_phis)
    
    return A_megalist, timestep_megalist, phi_list

def main():

    N, S, M, lambd = initiate_values()
    stability_period = 150000
    max_steps = 3e6
    n_runs = 20
    plot_phis_evo = False
    plot_final_phis = False

    A_megalist, timestep_megalist, phi_list = get_Alists_numba(N, S, M, lambd, n_runs=n_runs, max_steps=max_steps, period=stability_period)
    
    interesting_runs, max_timestep = [], 0
    fig, ax = plt.subplots()

    for n in range(len(A_megalist)):

        A_list, timesteps = A_megalist[n], timestep_megalist[n]
        if max(timesteps) > max_timestep: max_timestep = max(timesteps)
        if A_list[-1] < 0.9 * (S/M): interesting_runs.append(n)

        ax.plot(timesteps, A_list, 'grey', alpha=0.5)

    for t in np.arange(stability_period, max_timestep, stability_period):
    
        ax.axvline(t, 0, 1, lw=0.5, ls='dashed', color='k')
    
    ax.set_ylabel('A')
    ax.set_xlabel('timesteps')
    ax.set_title(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    plt.show()

    ### SEPARATE BIT IF THERE ARE MULTIPLE PHIS TO PLOT FOR THE SAME RUN

    if plot_phis_evo == True:

        phis_to_plot = len(phi_list[0])
        saved_timesteps = [2e5, 4e5, 6e5, 8e5, 1e6]

        for run in interesting_runs:

            fig, axs = plt.subplots(nrows=N, ncols=phis_to_plot, figsize=(phis_to_plot, int((N+1)*(S/M))), tight_layout=True)

            phis = phi_list[run]
            for i, phi_ in enumerate(phis):
                phi = np.array(phi_).reshape((N,S,M))
                for individual in range(N):
                    axs[individual, i].imshow(phi[individual,:,:], cmap='magma')
                    axs[individual, i].axis('off')
                    if individual==0: axs[individual, i].set_title(f'timestep\n{int(saved_timesteps[i])}', fontsize=8)
            fig.suptitle(f'final A = {A_megalist[run][-1]:.2f}')

        plt.show()

    ### CODE TO SHOW THE FINAL STATES OF ALL RUNS ENDING WITH A LOWER A-VALUE

    if plot_final_phis == True:

        runs_to_plot = len(interesting_runs)

        if runs_to_plot > 1:

            fig, axs = plt.subplots(nrows=N, ncols=runs_to_plot, figsize=(runs_to_plot, int((N+1)*(S/M))), tight_layout=True)

            for i, run in enumerate(interesting_runs):
                final_phi = np.array(phi_list[run][-1]).reshape((N,S,M))

                for individual in range(N):
                    axs[individual, i].imshow(final_phi[individual,:,:], cmap='magma')
                    axs[individual, i].axis('off')
                    if individual == 0: axs[individual, i].set_title(f'A = {A_megalist[run][-1]:.2f}', fontsize=8)

            plt.show()

        elif runs_to_plot == 1:

            fig, axs = plt.subplots(nrows=N, ncols=1, figsize=(1, int((N+1)*(S/M))), tight_layout=True)

            final_phi = np.array(phi_list[run][-1]).reshape((N,S,M))

            for individual in range(N):
                axs[individual].imshow(final_phi[individual,:,:], cmap='magma')
                axs[individual].axis('off')
                if individual == 0: axs[individual].set_title(f'A = {A_megalist[run][-1]:.2f}', fontsize=8)

            plt.show()


main()
