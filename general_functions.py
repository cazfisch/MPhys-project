import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import random
import os
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from numba import jit, njit
from numba.types import float64, int64

def initiate_values():

    global N, S, M
    if len(sys.argv) != 5:
        N, S, M = 2, 4, 4
        lambd = 0.01 # small-but-not-too-small
    else:
        N = int(sys.argv[1])
        S = int(sys.argv[2])
        M = int(sys.argv[3])
        lambd = float(sys.argv[4])
    
    return N, S, M, lambd

def choose_agents(N, method=None):

    # could define a PDF from which i and j are drawn

    if method == 'NN':
        speaker = np.random.randint(N)
        listener = np.random.choice([speaker-2, speaker-1, speaker+1, speaker+2], p=[0.1, 0.4, 0.4, 0.1])
        listener = listener%N

    else:
        speaker, listener = np.random.choice(N, size=2, replace=False)
        
    return speaker, listener
    
def choose_meaning(M, speaker=None, method=None, meaning_order=None, alpha=0.8):

    # could define a PDF dependent on the speaker i and the meaning intended m
    # intend = np.random.choice(M, p=rho(i,m))

    if method == 'zipf':
        meaning_ranks = np.array(meaning_order) + 1
        meaning_freqs = meaning_ranks ** (-alpha)
        meaning_freqs /= np.sum(meaning_freqs)
        intend = np.random.choice(M, p=meaning_freqs)
    
    else: intend = np.random.choice(M)
    
    return intend

def choose_signal(S, prob_array, speaker=None, intend=None, method=None, p_noise=0.05, alpha=0.8):

    if method == 'noise':
        signal = np.random.choice(S, p=prob_array)
        # add noise
        # small probability of signal confusion: 
            # 2 mechansims (identical): 
            # speaker produces wrong one by mistake -- listener might give positive feedback for wrong one
            # listener hears wrong one by mistake -- and again give positive feedback to the wrong mapping.
            # outcome is the same.

        if np.random.random() < p_noise: 
            new_list = list(range(S))
            new_list.pop(signal)
            signal = np.random.choice(new_list) # new signal, different to original one

    elif method == 'zipf':
        # makes the signals that require the least effort more likely to be chosen
        signal_ranks = np.arange(1, S+1) # unchanged order of signals, ranked by ease of use
        signal_freqs = signal_ranks ** (-alpha) # first signal is easiest to use, hence most frequent
        total_pdf = signal_freqs * prob_array # combine the inherent 'least-effort' probability with that given by conversations
        total_pdf /= np.sum(total_pdf) # normalise
        signal = np.random.choice(S, p=total_pdf)

    else: signal = np.random.choice(S, p=prob_array)

    return signal

def infer_meaning(M, prob_array, listener=None):

    return np.random.choice(M, p=prob_array)

def get_feedback(mu_pos, mu_neg, intend, infer, M, N=None, speaker=None, listener=None, method=None, social_ladder=None, f=50):

    # generally, chosen from distribution lambda_ij (m*, m')
    # could allow for near misses (depending on the meaning space)

    # positive feedback, with mu_pos > mu_neg

    feedback = mu_pos if infer == intend else mu_neg

    if method == 'leader_outcast':
        # all members want to mimic the leader 
        # all members want to do the opposite of the outcast
        # nobody listens to the rest of the community 

        if   listener == social_ladder[0]: # leader
            feedback *= 1
        elif listener == social_ladder[-1]: # outcast
            feedback *= -1
        else: feedback = 0

    elif method == 'linear_ladder':
        # increase or decrease the valuation of the listener's judgement 
        # based on their place on the social ladder

        judgement_factors = np.linspace(2, -1, N) # the scale is a bit arbitrary
        social_index = social_ladder.index(listener)
        feedback *= judgement_factors[social_index] 

    elif method == 'leader':

        if listener == social_ladder[0]: # leader
            feedback *= 10
        else: feedback *= 0.1

    elif method == 'outcast':
        
        if listener == social_ladder[-1]: # outcast
            feedback *= -1

    elif method == 'trendsetter':
        
        if listener != social_ladder[0]: # trendsetter
            feedback *= 1/f

    elif method == 'near_misses':
        # allow for positive feedback if the meaning intended is close to the one inferred
        # !! since the listener gives the feedback, they are the judge of whether the speaker is right or not.
        # only makes sense for a relatively large meaning space
        if intend == (infer - 1)%M or intend == (infer + 1)%M:
            feedback = mu_pos / 2 # arbitrary number -- less than mu+ and more than mu-
        
    return feedback

def change_social_ladder(ladder, method='swap'):
    
    if method=='swap':
        i1, i2 = random.sample(range(len(ladder)), 2)
        ladder[i1], ladder[i2] = ladder[i2], ladder[i1]
    
    elif method=='rotate_left':
        ladder = list(np.roll(ladder,-1)) # former social leader becomes outcast

    elif method=='rotate_right':
        ladder = list(np.roll(ladder, 1)) # former social outcast becomes leader

    return ladder

def U(phi, method=None):

    if method == 'linear':
        return phi
    else:
        return phi * (1 - phi)

def intelligibility(phi, N, S, M):

    psi = phi / phi.sum(axis=2)[:, :, np.newaxis]
    I = 0
    for n in range(1, N):
        I += 1/(N * (N-1) * M) * (phi * np.roll(psi, n, axis=0)).sum()
    
    return I

def equilibrium(phi, N, S, M, I=None, thresh_frac=0.99):
    max_val = (S / M) if M > S else 1
    if I is not None: 
        return I >= thresh_frac * max_val
    else:
        return intelligibility(phi, N, S, M) >= thresh_frac * max_val

def hesitation_finished(I, M, thresh_frac=0.3):
    return I > (1 + thresh_frac) / M

def initiate_phi(N, S, M, method=None, epsilon=0.5, diag_val=None):

    phi = np.ones((N,S,M)) * 1/S
    added_value = epsilon

    if method == 'hazy_diagonal':
        for i in range(N):
            for s in range(S):
                for m in range(M):
                    if s==m: phi[i,s,m] += added_value
                    else: phi[i,s,m] -= added_value/(S-1)

    elif method == 'diag_val':
        for i in range(N):
            for s in range(S):
                for m in range(M):
                    if s==m: phi[i,s,m] = diag_val
                    else: phi[i,s,m] = (1 - diag_val) / (S-1)
       
    return phi

def run_simple_model(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000, get_phi=False):

    # N, S, M, lambd = initiate_values()
    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps = [], []

    phi = initiate_phi(N, S, M)

    if get_phi == False:

        individual = np.random.choice(N)
        print(f'individual {individual}')

        fig, ax = plt.subplots()

        im = ax.imshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
        cbarticks = np.linspace(0,1,11)
        cbar = fig.colorbar(im, ax=ax, ticks=cbarticks)

    for i in range(max_steps):

        speaker, listener = choose_agents(N)
        intend = choose_meaning(M)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend])
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, speaker, listener, method=None, social_ladder=None)

        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if i % 1000 == 0 and i > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.5f}'.format(i, I))
            I_list.append(I)
            timesteps.append(i)

            if get_phi == False:

                plt.cla()
                ax.imshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
                plt.draw()
                plt.pause(0.0001)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=0.95): 
                print('equilibrium reached')
                break

    if get_phi == False:

        fig, axs = plt.subplots(ncols=N)
        for i, ax in enumerate(axs.flat):
            ax.imshow(phi[i,:,:], cmap='magma')
            ax.axis('off')
        fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

        plt.show()

    else: return phi

def run_model_nearmisses(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000):

    # N, S, M, lambd = initiate_values()
    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps = [], []

    phi = initiate_phi(N, S, M)
    individual = np.random.choice(N)
    print(f'individual {individual}')

    fig, ax = plt.subplots()

    im = ax.matshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
    # ax.set_yticks(range(S)); ax.set_xticks(range(M))
    cbarticks = np.linspace(0,1,11)
    cbar = fig.colorbar(im, ax=ax, ticks=cbarticks)

    for i in range(max_steps):

        intend = choose_meaning(M)
        speaker, listener = choose_agents(N)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend])
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, method='near_misses')

        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if i % 1000 == 0 and i > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.5f}'.format(i, I), end='\r')
            I_list.append(I)
            timesteps.append(i)

            plt.cla()
            ax.matshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.0001)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=0.95): 
                print('equilibrium reached')
                break

    fig, axs = plt.subplots(ncols=N)
    for i, ax in enumerate(axs.flat):
        ax.imshow(phi[i,:,:], cmap='magma')
        ax.axis('off')
    fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    plt.show()

    print(f'final intelligibility = {I_list[-1]:.3f}')

def run_model_noise(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000, p_noise=0.05):

    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps = [], []

    phi = initiate_phi(N, S, M)
    individual = np.random.choice(N)
    print(f'individual {individual}')

    fig, ax = plt.subplots()

    im = ax.matshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
    cbarticks = np.linspace(0,1,11)
    cbar = fig.colorbar(im, ax=ax, ticks=cbarticks)

    for i in range(max_steps):

        intend = choose_meaning(M)
        speaker, listener = choose_agents(N)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend], method='noise', p_noise=p_noise)
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, method=None)

        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if i % 5000 == 0 and i > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.5f}'.format(i, I))
            I_list.append(I)
            timesteps.append(i)

            plt.cla()
            ax.matshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.0001)

        if equilibrium(phi, N, S, M, thresh_frac=0.95): 
            print('equilibrium reached')
            break

    fig, axs = plt.subplots(ncols=N)
    for i, ax in enumerate(axs.flat):
        ax.imshow(phi[i,:,:], cmap='magma')
        ax.axis('off')
    fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    fig, ax = plt.subplots()
    ax.scatter(timesteps, I_list, marker='x')
    ax.set_ylabel('I')
    ax.set_xlabel('timesteps')
    ax.set_title(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    plt.show()

def run_model(max_steps=1_000_000):
    
    social_ladder = False

    # if social_ladder == True: 


    N, S, M, lambd = initiate_values()
    I_list, timesteps, phi_list, social_ladder_list, ladder_update_times = [], [], [], [], []
    social_ladder = list(range(N))
    print(social_ladder)

    phi = np.ones((N, S, M)) / S # phi = 1/S for all speakers
    individual = np.random.choice(N)
    print(f'individual {individual}')

    fig, ax = plt.subplots()

    im = ax.imshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
    cbarticks = np.linspace(0,1,11)
    cbar = fig.colorbar(im, ax=ax, ticks=cbarticks)

    for i in range(max_steps):

        intend = choose_meaning(M)
        speaker, listener = choose_agents(N, 'random')
        produce = np.random.choice(S, p=phi[speaker,:,intend]) # the signal produced by the speaker
        infer = np.random.choice(M, p=phi[listener,produce,:] / np.sum(phi[listener,produce,:])) # the meaning inferred by the listener

        # updating phi
        feedback = get_feedback(lambd, speaker, listener, intend, infer, method=None, social_ladder=None)
        phi[speaker,produce,intend] += (feedback * phi[speaker,produce,intend] * (1 - phi[speaker,produce,intend]))
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend]) # normalise along signal axis

        if i % 10000 == 0 and i > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.5f}'.format(i, I))
            I_list.append(I)
            timesteps.append(i)

            plt.cla()
            ax.imshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.0001)

        if i % 50000 == 0 and i > 0:
            phi_list.append(phi.copy())
            # social_ladder_list.append(social_ladder.copy())
            # ladder_update_times.append(i)
            # social_ladder = swap_random(social_ladder)
            # print(social_ladder)

        if equilibrium(phi, N, S, M, thresh_frac=0.95): 
            print('equilibrium reached')
            break

    fig, ax = plt.subplots()
    ax.scatter(timesteps, I_list, marker='x')
    ax.vlines(ladder_update_times, min(I_list), max(I_list), colors='k', alpha=0.5, lw=.5)
    ax.set_ylabel('I')
    ax.set_xlabel('timesteps')
    ax.set_title(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    fig, axs = plt.subplots(ncols=N)
    for i, ax in enumerate(axs.flat):
        ax.imshow(phi[i,:,:], cmap='magma')
        ax.axis('off')
    fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    ncols = min(10,len(phi_list))
    fig, axs = plt.subplots(nrows=N, ncols=ncols, figsize=(ncols, N+1))
    for i, ax in enumerate(axs.flat):
        ax.imshow(phi_list[i%ncols][i//ncols,:,:], cmap='magma')
        # if i//ncols == 0: ax.set_title(str(social_ladder_list[i]), fontsize=8)
        # elif i//ncols == N-1: ax.set_xlabel(f'timestep {i}', fontsize=8)
        ax.axis('off')
    fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    plt.show()

def optimal_grammar(N, S, M):

    phi_optimal = np.zeros((N,S,M))*1/S
    for s in range(S):
        for m in range(M):
            if s == m: phi_optimal[:, s, m] = 1 # along the diagonal -- only works for S=M

    return phi_optimal

def suboptimal_grammar(phi_optimal, N, S, M, epsilon=1e-3, method=None):

    phi_suboptimal = phi_optimal.copy()

    for n in range(N):
        for m in range(M):
            counter_s = 0
            if method == 'uniform': small_rand_values = np.ones(S-1)*epsilon
            else:                   small_rand_values = np.random.random(S-1)*epsilon
            for s in range(S):
                if phi_suboptimal[n,s,m] == 0: 
                    phi_suboptimal[n,s,m] += small_rand_values[counter_s]
                    counter_s += 1
                else: 
                    phi_suboptimal[n,s,m] += - np.sum(small_rand_values) 

    return phi_suboptimal

# Deterministic equations

def difference(phi_init, phi_final, N, S, M, normalised=False): 
    
    if normalised: 
        return np.sum(np.abs(phi_final - phi_init)) / (N*(S-1)*M)
    else: 
        return np.sum(np.abs(phi_final - phi_init))

def grid_display(phi, N, S, M, eps=None, with_title=False):
    
    cmap = 'OrRd'
    figsize = (5*(M/S)*(N/2),5)
    fig, axs = plt.subplots(ncols=N, figsize=figsize, sharey=True)
    if with_title: fig.subplots_adjust(top=0.8)

    axs[0].set_ylabel('S')
    for idx, ax in enumerate(axs):
        ax.matshow(phi[idx,:,:], cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel('M')
        for s in range(S):
            for m in range(M):
                ax.text(m, s, f'{phi[idx,s,m]:.2f}', ha='center', va='center')

    if with_title:
        if eps is not None: title = f'initial sub-optimal grammar $(\epsilon = {eps:.2f})$'
        else: title = 'initial sub-optimal grammar'
        fig.suptitle(title, y=0.75)
    plt.show()

def get_index(i, s, m, N, S, M):
    return i*M*S + s*M + m

def get_ism(index, N, S, M):
    i = index // (M*S)
    s = index % (M*S) // M
    m = index % (M*S) % M
    return i, s, m

def phi_to_y(phi, N, S, M):
    y = np.empty(N*S*M)
    for alpha in range(N*S*M):
        y[alpha] = phi[get_ism(alpha, N=N, S=S, M=M)]
    return y

def y_to_phi(y, N, S, M):
    phi = np.empty((N,S,M))
    for alpha in range(N*S*M):
        phi[get_ism(alpha, N=N, S=S, M=M)] = y[alpha]
    return phi

def deterministic_func(t, phi, mu, N, S, M):
    
    phi_dot = np.zeros(N*S*M)

    for alpha in np.arange(N*S*M):
        i,s,m = get_ism(alpha, N=N, S=S, M=M)
        sum_of_terms = 0

        for j in range(N):
            if j==i: continue # omit the j=i case

            for s_ in range(S):
                delta_ss = 1 if s_==s else 0
                sum_denominator = 0
                for m_ in range(M): sum_denominator += phi[get_index(j, s_, m_, N=N, S=S, M=M)]

                sum_of_terms += mu/(N*(N-1)*M) * phi[get_index(i,s_,m, N=N, S=S, M=M)]**2 * (1 - phi[get_index(i,s_,m, N=N, S=S, M=M)]) * \
                                (2 * phi[get_index(j,s_,m, N=N, S=S, M=M)] / sum_denominator - 1) * (delta_ss - phi[alpha])

        phi_dot[alpha] = sum_of_terms 
    
    return phi_dot

def get_deterministic_predictions(N, S, M, lambd, tf=100000, eval_step=1000, phi_initial=None, get_intelligibility=True):

    phi_init = initiate_phi(N, S, M, method='hazy_diagonal', diag_val=1/S+0.01)
    if phi_initial is not None: phi_init = phi_initial
    y0 = phi_to_y(phi_init, N=N, S=S, M=M)
    t_eval = np.arange(0, tf+eval_step, eval_step)
    sol = solve_ivp(deterministic_func, t_span=[0, tf], t_eval=t_eval, y0=y0, method='RK45', vectorized=True, args=(lambd, N, S, M))
    
    if get_intelligibility: 

        Is = []
        for t in range(len(t_eval)):
            y = sol.y[:,t]
            phi = y_to_phi(y, N=N, S=S, M=M)
            I = intelligibility(phi, N, S, M)
            Is.append(I)

        return t_eval, Is

    else:

        phis = []
        for t in range(len(t_eval)):
            y = sol.y[:,t]
            phi = y_to_phi(y, N=N, S=S, M=M)
            phis.append(phi)

        return t_eval, phis
        
# Linear stability analysis

def constraint_satisfied(vr, N, S, M, tolerance=1e-10):

    # sum over s
    tolerated = True
    for i in range(N):
        for m in range(M): # for each m cumulate the coefficients across all s
            sum_s = 0
            for s in range(S):
                idx = get_index(i, s, m, N, S, M)
                sum_s += vr[idx]
            # if one doesn't satisfy the constraint, the whole vector is rejected
            if not (-tolerance < sum_s < tolerance): tolerated = False 
    
    return tolerated

def null_coeff(vl, N, S, M, tolerance=1e-10, upper_bound=1e-3):

    # generate epsilon_0, knowing sum_s of epsilon(i,s,m) = 0

    epsilon_0 = np.zeros(N*S*M)

    for i in range(N):
        for m in range(M):
            sum_s = 0
            for s in range(S):
                alpha = get_index(i, s, m, N, S, M)
                if s != S-1: 
                    small_rand_value = np.random.random()*upper_bound
                    epsilon_0[alpha] = small_rand_value
                    sum_s += small_rand_value
                else: 
                    epsilon_0[alpha] = - sum_s

    # verify whether a_i = vl_i dot epsilon_0 gives zero, i.e. if the coefficient a_i disappears

    if np.abs(np.dot(epsilon_0, vl)) < tolerance: return True
    else: return False

def evecs_display(vector, lmbda, deg_index, N, S, M, mu, type='right', savefig=False, with_title=False, colormap=None, with_text=False):

    plots_dir = 'plots/linear stability/'
    case_dir = f'({N}.{S}.{M}.{mu})/'
    if not os.path.exists(plots_dir+case_dir): os.makedirs(plots_dir+case_dir)

    cmap = 'summer' if type == 'left' else 'Wistia'
    if colormap is not None: cmap = colormap

    if type in ['left', 'right', 'dummy']:

        grid = np.zeros((N,S,M))
        for idx, v in enumerate(vector): grid[get_ism(idx, N, S, M)] = v.real
        
        figsize = (5, 2) #(5*(M/S)*(N/2),5)
        fig, axs = plt.subplots(ncols=N, figsize=figsize, sharey=True)

        if with_title:
            title = f'$\lambda_d = {lmbda: .4f} \ ({deg_index})$'
            if type=='dummy': title = 'dummy initial vector'
            fig.suptitle(title, y=0.75)
            fig.subplots_adjust(top=0.8)

        axs[0].set_ylabel('S')
        for idx, ax in enumerate(axs):
            im = ax.matshow(grid[idx,:,:], cmap=cmap, vmin=np.min(grid), vmax=np.max(grid))
            ax.set_xlabel('M')
            if with_text:
                for s in range(S):
                    for m in range(M):
                        ax.text(m, s, f'{grid[idx,s,m]:.2f}', ha='center', va='center')

        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.12, 0.03, 0.75])
        fig.colorbar(im, cax=cax)

        if savefig: 
            if abs(lmbda) < 1e-15: _lmbda = abs(lmbda) # avoid the zero case
            else: _lmbda = lmbda
            fig.savefig(plots_dir+case_dir+f'{_lmbda:.4f}({deg_index})-report.png')
        
        plt.show()

    elif type == 'left_right':

        vl, vr = vector
        grid_l = np.zeros((N,S,M))
        grid_r = np.zeros((N,S,M))
        for idx, v in enumerate(vl): grid_l[get_ism(idx, N, S, M)] = v.real
        for idx, v in enumerate(vr): grid_r[get_ism(idx, N, S, M)] = v.real

        fig = plt.figure(constrained_layout=True, figsize=(int(4*M/S), 5)) 
        suptitle = f'$\lambda_d = {lmbda: .4f} \ ({deg_index})$'
        if null_coeff(vl, N, S, M): suptitle = f'$\lambda_d = {lmbda: .4f} \ ({deg_index})$ / vanishing coefficient'
        fig.suptitle(suptitle)

        subfigs = fig.subfigures(nrows=2, ncols=1)
        for row, subfig in enumerate(subfigs):

            if row == 0:
                title = 'Right eigenvector'
                grid = grid_r
                cmap = 'Wistia'
            else:
                title = 'Left eigenvector'
                grid = grid_l
                cmap = 'summer'

            subfig.suptitle(title, x=0.1, ha='left')
            # subfig.subplots_adjust(top=0.8)

            axs = subfig.subplots(nrows=1, ncols=2)
            for idx, ax in enumerate(axs):
                ax.matshow(grid[idx,:,:], cmap=cmap)
                if idx==0: ax.set_ylabel('S')
                ax.set_xlabel('M')
                for s in range(S):
                    for m in range(M):
                        ax.text(m, s, f'{grid[idx,s,m]:.2f}', ha='center', va='center')

        if savefig: 
            if abs(lmbda) < 1e-15: _lmbda = abs(lmbda) # avoid the zero case
            else: _lmbda = lmbda
            fig.savefig(plots_dir+case_dir+f'{_lmbda:.4f}({deg_index})-report.png')
        
        plt.show()

def A_matrix(phi, N, S, M, mu):

    # implementing Eq. 27 from the 'symmetric fixed point' notes

    # generating phi matrix
    psi = np.zeros((N,S,M))
    for n in range(N):
        for s in range(S):
            denominator = np.sum(phi[n,s,:])
            for m in range(M):
                psi[n,s,m] = phi[n,s,m] / denominator

    # simplest case for rho, G_ij, lambda and U[{phi}]

    def rho(n, m): return 1/M
    def G_nn_(n,n_): return 1/(N*(N-1)) if n!=n_ else 0
    def lambd(n, n_, m, m_, mu=mu): return mu if m==m_ else -mu
    def U(phi_nsm): return phi_nsm * (1 - phi_nsm)
    def U_deriv(phi_nsm): return 1 - 2*phi_nsm

    A = np.zeros((N*S*M, N*S*M))

    for alpha in range(N*S*M):
        n,s,m = get_ism(alpha, N, S, M)

        for alpha_ in range(N*S*M):
            n_,s_,m_ = get_ism(alpha_, N, S, M) # corresponding to n', s' and m'
            
            delta_nn = 1 if n==n_ else 0
            delta_ss = 1 if s==s_ else 0
            delta_mm = 1 if m==m_ else 0

            # first of three terms
            first_term = 0

            for s_hat in range(S):
            
                # n^, m^ sum
                sum_nm = 0
                for n_hat in range(N):
                    for m_hat in range(M):
                        sum_nm += G_nn_(n, n_hat) * lambd(n, n_hat, m, m_hat) * psi[n_hat, s_hat, m_hat]
                
                first_term += phi[n, s_hat, m] * U(phi[n, s_hat, m]) * sum_nm
            
            first_term *= - delta_nn * delta_ss * delta_mm

            # second of three terms

            sum_nm = 0
            for n_hat in range(N):
                for m_hat in range(M):
                    sum_nm += G_nn_(n, n_hat) * lambd(n, n_hat, m, m_hat) * psi[n_hat, s_, m_hat]
            
            second_term = delta_nn * delta_mm * (delta_ss - phi[n,s,m]) * (U(phi[n,s_,m]) + phi[n,s_,m] * U_deriv(phi[n,s_,m])) * sum_nm

            # third of three terms

            denom = np.sum(phi[n_,s_,:])
            numerator = 0
            for m_hat in range(M):
                numerator += lambd(n, n_, m, m_hat) * phi[n_, s_, m_hat]

            third_term = (delta_ss - phi[n,s,m]) * phi[n,s_,m] * U(phi[n,s_,m]) * G_nn_(n,n_) / denom * \
                         (lambd(n, n_, m, m_) - numerator / denom)
            
            A[alpha, alpha_] = rho(n, m) * ( first_term + second_term + third_term )

    return A

def eigen_vals_vecs(phi_fixed, N, S, M, mu, show_evecs=False, savefigs=False, colormap=None, unphysical=False):

    A = A_matrix(phi_fixed, N, S, M, mu)

    w, v = eig(A)

    sorted_inds = np.argsort(-np.abs(w)) # sorted from largest to smallest abs value

    w_sorted = w[sorted_inds]
    v_sorted = v[:,sorted_inds]

    print(f'configuration (N, S, M, lambda) = ({N}, {S}, {M}, {mu})')
    print('----------------------------------------------------------------------------')

    for i in range(w.size):

        constraint_respected = '*' if constraint_satisfied(v_sorted[:,i], N, S, M) else 'constraint not satisfied'
        vanishornot = 'coef vanishes!' if null_coeff(v_sorted[:,i], N, S, M) else '*'
        print('w = {: .2e} /// {} - {}'.format(w_sorted[i].real, constraint_respected, vanishornot))

    if show_evecs:

        deg_index = 1
        prev_w = 0

        for i in range(w.size):

            if unphysical:

                if (not constraint_satisfied(v_sorted[:,i], N, S, M) and null_coeff(v_sorted[:,i], N, S, M)):

                    lmbda = w_sorted[i].real

                    if abs(prev_w - lmbda) > 1e-10: deg_index = 1
                    evecs_display(v_sorted[:,i], lmbda, deg_index, N, S, M, mu, type='right', savefig=savefigs, colormap=colormap)
                    deg_index += 1
                    prev_w = lmbda
            else:
            
                if (constraint_satisfied(v_sorted[:,i], N, S, M) and not null_coeff(v_sorted[:,i], N, S, M)):

                    lmbda = w_sorted[i].real

                    if abs(prev_w - lmbda) > 1e-10: deg_index = 1
                    evecs_display(v_sorted[:,i], lmbda, deg_index, N, S, M, mu, type='right', savefig=savefigs, colormap=colormap)
                    deg_index += 1
                    prev_w = lmbda

# Numba functions

@njit
def rand_choice_numba(prob):
    return np.searchsorted(np.cumsum(prob), np.random.random(), side='right')

@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def round_numba(x, decimals, out):
    return np.round_(x, decimals, out)

@njit
def intelligibility_numba(phi, N, S, M):

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
def equilibrium_numba(phi, N, S, M, threshold=0.95, decision=False, decision_frac=0.3):

    I = intelligibility_numba(phi, N, S, M)
    if decision: return I > 1/M * (1 + decision_frac)
    else:
        if M >= S: return (I > threshold * (S/M))
        else: return (I > threshold)

