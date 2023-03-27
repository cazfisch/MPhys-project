from general_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from numba import njit

@njit
def run_model_trendsetter_numba(N=3, S=5, M=5, lambd=0.01, f=10, wait_time=100000, max_steps=500000, eq=False, diag_val=0.8):
     
    # initialise to hazy diagonal
    phi = np.ones((N,S,M)) * 1/S
    for i in range(N):
        for s in range(S):
            for m in range(M):
                if s==m: phi[i,s,m] = diag_val
                else: phi[i,s,m] = (1 - diag_val) / (S-1)

    trendsetter = 0
    swap_timesteps, I_swaps = [], []
    save_phis = []
    I_list, timesteps = [], []

    for step in range(max_steps):

        if step % wait_time == 1: 
            swap_timesteps.append(step); I_swaps.append(intelligibility_numba(phi, N, S, M))
            m1 = np.random.randint(M)
            m2 = np.random.randint(M)
            while m2 == m1: m2 = np.random.randint(M)
            s1, s2 = np.argmax(phi[trendsetter, :, m1]), np.argmax(phi[trendsetter, :, m2])

            s1_m1 = phi[trendsetter, s1, m1]
            s2_m1 = phi[trendsetter, s2, m1]
            phi[trendsetter, s1, m1] = s2_m1
            phi[trendsetter, s2, m1] = s1_m1

            s1_m2 = phi[trendsetter, s1, m2]
            s2_m2 = phi[trendsetter, s2, m2]
            phi[trendsetter, s1, m2] = s2_m2
            phi[trendsetter, s2, m2] = s1_m2

        # update rule
        intend = np.random.randint(M) # meaning intended -- rho = 1/M, uniform distribution
        speaker = np.random.randint(N)
        listener = np.random.randint(N)
        while listener == speaker: listener = np.random.randint(N)
        produce = rand_choice_numba(prob = phi[speaker,:,intend])
        infer = rand_choice_numba(prob = phi[listener,produce,:] / np.sum(phi[listener,produce,:]))
        feedback = lambd if infer == intend else - lambd
        if listener == trendsetter:
            feedback *= f
        phi[speaker,produce,intend] += (feedback * phi[speaker,produce,intend] * (1 - phi[speaker,produce,intend])) # * phi[speaker,produce,intend] * (1 - phi[speaker,produce,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend]) # normalise along signal axis

        # save phis
        if (step % wait_time == 0 or step == max_steps-1):
            save_phis.append(phi.copy())
        
        # record intelligibility
        if step % (max_steps/200) == 0 and step > 0:
            I = intelligibility_numba(phi, N, S, M)
            I_list.append(I)
            timesteps.append(step)
            if equilibrium_numba(phi, N, S, M, threshold=0.99) and eq==True: 
                print('equilibrium reached')
                break

    return timesteps, I_list, swap_timesteps, save_phis, I_swaps
    
def plot_results(timesteps, I_list, swap_timesteps, save_phis, N=3, S=5, M=5, lambd=0.01, f=10, wait_time=100000, with_swaps=True, save=False):

    # plot the intelligibility
    fontsize=12
    fig, ax = plt.subplots(figsize=(7,3), tight_layout=True)
    ax.plot(timesteps, I_list, c='teal', alpha=0.4, lw=3)
    if with_swaps:
        for timestep in swap_timesteps: ax.axvline(timestep, 0, 1, color='k', alpha=0.5, lw=.5)
    ax.set_ylabel('intelligibility', fontsize=fontsize)
    ax.set_xlabel('conversations', fontsize=fontsize)
    if save: fig.savefig('/Users/casimirfisch/Desktop/Uni/MPhys/code/plots/language change/waittime20000_f10_test/'+f'int.png', dpi=300)

    if with_swaps:
        for idx, phi in enumerate(save_phis):
            fig, axs = plt.subplots(ncols=N, figsize=(N*2,3), tight_layout=True)
            for i in range(N): 
                axs[i].imshow(phi[i,:,:], interpolation='nearest', cmap='magma', vmin=0, vmax=1)
                if i == 0: axs[i].set_title('trendsetter')
                else:                axs[i].set_title('')
                axs[i].axis('off')
            if save: fig.savefig('/Users/casimirfisch/Desktop/Uni/MPhys/code/plots/language change/waittime20000_f10_test/'+f'phi_{idx}.png', dpi=300)

    plt.show()

def run_plot_nruns(N, S, M, lambd, f, wait_time, max_steps, diag_val, n_runs, with_vlines=True):

    fig, ax = plt.subplots()
    
    ax.set_ylabel('intelligibility')
    ax.set_xlabel('conversations')

    colours = list(mcolors.TABLEAU_COLORS.keys())[:n_runs]

    for i in range(n_runs):
        print(f'run {i}')
        timesteps, I_list, swap_timesteps, _, I_swaps = run_model_trendsetter_numba(N, S, M, lambd, f, wait_time, max_steps, diag_val=diag_val)
        if i==0 and with_vlines:
            for timestep in swap_timesteps: ax.axvline(timestep, 0, 1, color='k', alpha=0.5, lw=.5)
        ax.plot(timesteps, I_list, alpha=0.1, color=colours[i])
        ax.plot(swap_timesteps, I_swaps, marker='o', markersize=6, alpha=0.5, color=colours[i], label=f'run {i}')

    ax.legend()  
    plt.show()

def vary_wait_time(N, S, M, lambd, f, wait_times, max_steps, with_averages=False, n_avgd_runs=100):

    fontsize = 12
    n_runs = len(wait_times)
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_ylabel('intelligibility', fontsize=fontsize)
    ax.set_xlabel('conversations', fontsize=fontsize)
    colours = ['steelblue', 'darkred', 'teal', 'darkorchid']  # list(mcolors.TABLEAU_COLORS.keys())[:n_runs]

    for i, wait_time in enumerate(wait_times):
        print(f'wait time = {wait_time}')

        if with_averages: 

            I_megalist = []

            for j in range(50):
                timesteps, I_list, _, _, _ = run_model_trendsetter_numba(N, S, M, lambd, f, wait_time, max_steps)
                I_megalist.append(I_list)

            Is = np.array(I_megalist)
            mean_I = np.mean(Is, axis=0)
            err_I  = np.std(Is, axis=0) / np.sqrt(n_avgd_runs)
            ax.plot(timesteps, mean_I, alpha=0.5, color=colours[i], label=f'$T_w = {wait_time}$', lw=2)
        
        else:

            timesteps, I_list, _, _, _ = run_model_trendsetter_numba(N, S, M, lambd, f, wait_time, max_steps)
            ax.plot(timesteps, I_list, alpha=0.5, color=colours[i], label=f'wait time = {wait_time}')

    ax.legend(frameon=False)
    fig.savefig('/Users/casimirfisch/Desktop/Uni/MPhys/code/plots/language change/' + f'wait_times{wait_times}_alpha{f}.png', dpi=400)  
    plt.show()
    

def main():

    N, S, M, lambd, f, wait_time, max_steps, diag_val = 3, 5, 5, 0.01, 10, 75000, 1000000, 0.8

    # timesteps, I_list, swap_timesteps, save_phis, _ = run_model_trendsetter_numba(N, S, M, lambd, f, wait_time, max_steps, diag_val=diag_val)
    # plot_results(timesteps, I_list, swap_timesteps, save_phis, N, S, M, lambd, f, wait_time, with_swaps=False)

    run_plot_nruns(N, S, M, lambd, f, wait_time, max_steps, diag_val, n_runs=10, with_vlines=False)

    # vary_wait_time(N, S, M, lambd, f, [500, 2000, 20000, 100000], max_steps, with_averages=True)

main()