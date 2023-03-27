import numpy as np
import matplotlib.pyplot as plt
from general_functions import *
import time

def run_simple_model(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000, init=None, eps=None, update_step=1000, show_I=False, thresh_frac=0.95):

    # N, S, M, lambd = initiate_values()
    mu_pos, mu_neg = lambd, -lambd
    if show_I: I_list, timesteps = [], []

    if init == 'sub_opt':
        phi = suboptimal_grammar(optimal_grammar(N, S, M), N, S, M, epsilon=eps, method='uniform') 
    else: 
        phi = initiate_phi(N, S, M)

    # individual = np.random.choice(N)
    # print(f'individual {individual}')

    # fig, ax = plt.subplots()

    # im = ax.imshow(phi[individual,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
    # cbarticks = np.linspace(0,1,11)
    # cbar = fig.colorbar(im, ax=ax, ticks=cbarticks)

    fig, axs = plt.subplots(ncols=N, figsize=(N*2,3))
    for i in range(N): 
        axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
        # axs[i].axis('off')

    for i in range(max_steps):

        speaker, listener = choose_agents(N)
        intend = choose_meaning(M)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend])
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M)

        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if i % update_step == 0 and i > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.5f}'.format(i, I))
            if show_I:
                I_list.append(I)
                timesteps.append(i)

            plt.cla()
            for i in range(N): axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.0001)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=thresh_frac): 
                print('equilibrium reached')
                break

    # fig, axs = plt.subplots(ncols=N)
    # for i, ax in enumerate(axs.flat):
    #     ax.imshow(phi[i,:,:], cmap='magma')
    #     ax.axis('off')
    # fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    if show_I: 

        fig, ax = plt.subplots()
        ax.plot(timesteps, I_list, marker='x')
        ax.set_ylabel('intelligibility')
        ax.set_xlabel('timesteps')
        ax.set_title(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

        plt.show()

def run_model_nearmisses(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000):

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

def run_model_social_ladder(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000, period=10000):

    ladder_method = 'leader_outcast'
    # methods so far:
    # leader_outcast: a leader and an outcast, with feedbacks multiplied by 1(2) and -1
    # linear_ladder : judgement factors associated with the social ladder -- quite arbitrary, especially when negatives are involved
    # leader        : only a leader -- experimenting with feedback factors: if lambda = 0.01, then leader = 0.1 while the rest can be lower (0.001)
    # outcast       : only an outcast

    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps, saved_phis, ladder_update_times = [], [], [], []
    social_ladder = list(range(N))

    phi = initiate_phi(N, S, M)

    # initialise animation
    fig, axs = plt.subplots(ncols=N)
    for i in range(N): 
        axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
        if ladder_method == 'leader':
            if i == social_ladder[0]: axs[i].set_title('leader')
            else:                      axs[i].set_title('')
        elif ladder_method == 'outcast':
            if i == social_ladder[-1]: axs[i].set_title('outcast')
            else:                      axs[i].set_title('')
        elif ladder_method == 'leader_outcast':
            if   i ==  social_ladder[0]: axs[i].set_title('leader')
            elif i == social_ladder[-1]: axs[i].set_title('outcast')
            else:                        axs[i].set_title('')

    for step in range(max_steps):

        speaker, listener = choose_agents(N)
        intend = choose_meaning(M)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend])
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, N, speaker, listener, method=ladder_method, social_ladder=social_ladder)

        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if step % 1000 == 0 and i > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.3f}'.format(step, I))
            I_list.append(I)
            timesteps.append(step)

            # animation
            plt.cla()
            for i in range(N): 
                axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
                if ladder_method == 'leader':
                    if i == social_ladder[0]: axs[i].set_title('leader')
                    else:                      axs[i].set_title('')
                elif ladder_method == 'outcast':
                    if i == social_ladder[-1]: axs[i].set_title('outcast')
                    else:                      axs[i].set_title('')
                elif ladder_method == 'leader_outcast':
                    if   i ==  social_ladder[0]: axs[i].set_title('leader')
                    elif i == social_ladder[-1]: axs[i].set_title('outcast')
                    else:                        axs[i].set_title('')
                
            plt.draw()
            plt.pause(0.0001)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=0.95): 
                print('equilibrium reached')
                break
        
        # change social ladder
        if step % period == 0 and step > 0:
            social_ladder = change_social_ladder(social_ladder, 'rotate_left')
            ladder_update_times.append(step)
            saved_phis.append(list(phi.flatten()))

    # plot the intelligibility
    fig, ax = plt.subplots()
    ax.plot(timesteps, I_list, marker='x')
    for ladder_time in ladder_update_times:
        ax.axvline(ladder_time, 0, 1, color='k', alpha=0.5, lw=.5)
    ax.set_ylabel('intelligibility')
    ax.set_xlabel('timesteps')
    ax.set_title(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    # plot evolution of phis, everytime the social ladder changes
    if len(saved_phis) != 0:
        ncols = min(10, len(saved_phis))
        fig, axs = plt.subplots(nrows=N, ncols=ncols, figsize=(ncols, N+1))
        for i, ax in enumerate(axs.flat):
            phi = np.array(saved_phis[i%ncols]).reshape((N,S,M))
            ax.imshow(phi[i//ncols,:,:], cmap='magma')
            ax.axis('off')
        fig.suptitle(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    plt.show()

def run_model_trendsetter(N=2, S=3, M=3, lambd=0.01, f=50, max_steps=1_000_000, wait_time=200, threshold=0.8, anim=True, anim_step=50, eq=True):

    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps = [], []
    social_ladder = list(range(N))

    trendsetter = social_ladder[0]

    phi = initiate_phi(N, S, M, method='diag_val', diag_val=0.8)

    if anim:

        # initialise animation
        fig, axs = plt.subplots(ncols=N, figsize=(N*2,3))
        fig.suptitle(f'$\lambda = {lambd}$, f={f}, wait time = {wait_time}')
        for i in range(N): 
            axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            if i == trendsetter: axs[i].set_title('trendsetter')
            else:                axs[i].set_title('')

    for step in range(max_steps):

        if step % wait_time == 0: # wait time is over, trendsetter swaps two mappings

            # randomly choose two signals to swap above a threshold
            signals, meanings = np.where(phi[trendsetter] >= threshold)
            n_mappings = len(signals)
            # only do this there are at least 2 mappings above the threshold
            if n_mappings >= 2:
                idx1, idx2 = np.random.choice(n_mappings, size=2, replace=False)
                # out of the mappings above the thresh, these are randomly selected
                s1, s2 = signals[idx1], signals[idx2]
                m1, m2 = meanings[idx1], meanings[idx2]
                # swap them in phi 
                phi[trendsetter, [s1, s2], m1] = phi[trendsetter, [s2, s1], m1]
                phi[trendsetter, [s1, s2], m2] = phi[trendsetter, [s2, s1], m2]

    
        # update rule as normal
        speaker, listener = choose_agents(N)
        intend = choose_meaning(M)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend])       
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, N, speaker, listener, method='trendsetter', social_ladder=social_ladder, f=50)
        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if step % anim_step == 0 and step > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.3f}'.format(step, I))
            I_list.append(I)
            timesteps.append(step)

            if anim:
                plt.cla()
                for i in range(N): axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
                plt.draw()
                plt.pause(0.0001)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=0.99) and eq==True: 
                print('equilibrium reached')
                break

    # plot the intelligibility
    fig, ax = plt.subplots()
    ax.scatter(timesteps, I_list, marker='x')
    ax.set_ylabel('intelligibility')
    ax.set_xlabel('timesteps')
    ax.set_title(f'$(N, S, M, \lambda) = ({N}, {S}, {M}, {lambd})$')

    plt.show()

def run_model_trendsetter_nothresh(N=2, S=3, M=3, lambd=0.01, f=50, max_steps=1_000_000, wait_time=200, anim=True, anim_step=50, eq=True, save_phis=False):

    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps = [], []
    swap_timesteps = []
    social_ladder = list(range(N))

    trendsetter = social_ladder[0]

    phi = initiate_phi(N, S, M, method='diag_val', diag_val=0.8)

    if anim:

        # initialise animation
        fig, axs = plt.subplots(ncols=N, figsize=(N*2,3))
        fig.suptitle(f'$\lambda = {lambd}$, f={f}, wait time = {wait_time}')
        for i in range(N): 
            axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            if i == trendsetter: axs[i].set_title('trendsetter')
            else:                axs[i].set_title('')
            axs[i].axis('off')

    for step in range(max_steps):

        if step % wait_time == 1: # wait time is over, trendsetter swaps two mappings

            swap_timesteps.append(step)

            m1, m2 = np.random.choice(M, size=2, replace=False)
            s1, s2 = np.argmax(phi[trendsetter, :, m1]), np.argmax(phi[trendsetter, :, m2])

            phi[trendsetter, [s1, s2], m1] = phi[trendsetter, [s2, s1], m1]
            phi[trendsetter, [s1, s2], m2] = phi[trendsetter, [s2, s1], m2]
    
        # update rule as normal
        speaker, listener = choose_agents(N)
        intend = choose_meaning(M)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend])       
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, N, speaker, listener, method='trendsetter', social_ladder=social_ladder, f=f)
        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if (step % wait_time == 0 or step == max_steps-1) and anim: # anim_step) - (anim_step-1)

            plt.cla()
            for i in range(N): axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
            plt.draw()
            plt.pause(0.0001)
        
        # intelligibility
        if step % (max_steps/100) == 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.3f}'.format(step, I))
            I_list.append(I)
            timesteps.append(step)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=0.99) and eq==True: 
                print('equilibrium reached')
                break

    # plot the intelligibility
    fig, ax = plt.subplots()
    ax.plot(timesteps, I_list)
    for timestep in swap_timesteps:
        ax.axvline(timestep, 0, 1, color='k', alpha=0.5, lw=.5)
    ax.set_ylabel('intelligibility')
    ax.set_xlabel('conversations')
    ax.set_title(f'$(N, S, M, \lambda, f, T_w) = ({N}, {S}, {M}, {lambd}, {f}, {wait_time})$')

    plt.show()

def run_model_zipf(N=2, S=3, M=3, lambd=0.01, max_steps=1_000_000, alpha=0.8, wait_time=10000, anim=True):

    meaning_order = np.arange(M)
    # np.random.shuffle(meaning_order)
    # print(meaning_order)

    mu_pos, mu_neg = lambd, -lambd
    I_list, timesteps = [], []

    phi = initiate_phi(N, S, M) # method='hazy_diagonal', epsilon=0.5

    if anim:

        # initialise animation
        fig, axs = plt.subplots(ncols=N, figsize=(N*2,3))
        fig.suptitle(f'Zipf\'s law')
        for i in range(N): 
            axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)

    for step in range(max_steps):

        speaker, listener = choose_agents(N)
        intend = choose_meaning(M, method='zipf', alpha=alpha, meaning_order=meaning_order)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend], method='zipf', alpha=0.5)      
        infer = infer_meaning(M, prob_array=phi[listener,signal,:] / np.sum(phi[listener,signal,:]))
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M)

        phi[speaker,signal,intend] += feedback * U(phi[speaker,signal,intend])
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])

        if step % 5000 == 0 and step > 0:

            I = intelligibility(phi, N, S, M)
            print('timestep {:7d} -- I = {:.3f}'.format(step, I))
            I_list.append(I)
            timesteps.append(step)

            if anim:

                # animation
                plt.cla()
                for i in range(N): axs[i].imshow(phi[i,:,:], interpolation='nearest', animated=True, cmap='magma', vmin=0, vmax=1)
                plt.draw()
                plt.pause(0.0001)

            if equilibrium(phi, N, S, M, I=I, thresh_frac=0.95): 
                print('equilibrium reached')
                break

        if step % wait_time == 0 and step > 0:

            np.random.shuffle(meaning_order)
            print(meaning_order)

    # plot the intelligibility
    fig, ax = plt.subplots()
    ax.plot(timesteps, I_list, marker='x')
    ax.set_ylabel('intelligibility')
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

def main(N=2, S=4, M=4, lambd=0.1, method=None, init=None):

    max_steps = 200000

    time.sleep(3) # allows to record from the initial state

    if   method == 'near_misses':
        run_model_nearmisses(N=N, S=S, M=M, lambd=lambd, max_steps=max_steps)
    elif method == 'social_ladder':
        run_model_social_ladder(N=N, S=S, M=M, lambd=lambd, max_steps=max_steps, period=10000)
    elif method == 'noise':
        run_model_noise(N=N, S=S, M=M, lambd=lambd, max_steps=max_steps, p_noise=0.5)
    elif method == 'trendsetter':
        run_model_trendsetter_nothresh(N=N, S=S, M=M, lambd=lambd, f=10, max_steps=200000, wait_time=50000, anim=True, anim_step=20000, eq=False)
    elif method == 'zipf':
        run_model_zipf(N=N, S=S, M=M, lambd=lambd, max_steps=max_steps, wait_time=20000, alpha=2., anim=True)
    else:
        run_simple_model(N=N, S=S, M=M, lambd=lambd, max_steps=max_steps, init=None, eps=1/3, update_step=2000, show_I=True, thresh_frac=0.95)

    # try something else: lowest effort signals, biasing production rule + changing frequencies of meanings over time

main(N=2, S=7, M=7, lambd=0.1, method=None)
