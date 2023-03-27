import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from general_functions import *

def func(frame, img, phi, N, S, M, conversations, mu_pos, mu_neg):

    for i in range(int(conversations)):

        speaker, listener = choose_agents(N)
        intend = choose_meaning(M, speaker=speaker)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend], speaker=speaker, intend=intend)
        infer = infer_meaning(M, phi[listener,signal,:] / np.sum(phi[listener,signal,:]), listener=listener)
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, speaker=speaker, listener=listener, method='near_misses')

        phi[speaker,signal,intend] += (feedback * U(phi[speaker,signal,intend]))
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])
    
    img.set_data(phi[0])
    return img,

def func_nearmisses(frame, img, phi, N, S, M, conversations, mu_pos, mu_neg):

    for i in range(int(conversations)):

        speaker, listener = choose_agents(N)
        intend = choose_meaning(M, speaker=speaker)
        signal = choose_signal(S, prob_array=phi[speaker,:,intend], speaker=speaker, intend=intend)
        infer = infer_meaning(M, phi[listener,signal,:] / np.sum(phi[listener,signal,:]), listener=listener)
        feedback = get_feedback(mu_pos, mu_neg, intend, infer, M, speaker=speaker, listener=listener, method='near_misses')

        phi[speaker,signal,intend] += (feedback * U(phi[speaker,signal,intend]))
        phi[speaker,:,intend] /= np.sum(phi[speaker,:,intend])
    
    img.set_data(phi[0])
    return img,

def good_animation_values(NSM):

    if NSM == (2, 3, 3):

        N, S, M = NSM
        # conversations, mu_pos, mu_neg = 1000, 2e-2, 1e-2
        conversations, mu_pos, mu_neg = 1000, 1e-2, -1e-2

    elif NSM == (2, 4, 4):

        N, S, M = NSM
        # conversations, mu_pos, mu_neg = 1000, 2e-2, 1e-2
        conversations, mu_pos, mu_neg = 2000, 1e-2, -1e-2

    elif NSM == (2, 5, 5):

        N, S, M = NSM
        # conversations, mu_pos, mu_neg = 1000, 2e-2, 1e-2
        conversations, mu_pos, mu_neg = 4000, 1e-2, -1e-2

    elif NSM == (4, 5, 5):

        N, S, M = NSM
        conversations, mu_pos, mu_neg = 12000, 1e-2, -1e-2

    elif NSM == (2, 6, 6):

        N, S, M = NSM
        conversations, mu_pos, mu_neg = 5000, 1e-2, -1e-2

    return N, S, M, conversations, mu_pos, mu_neg

def run_animation(method=None, show_final=False):

    writergif = animation.PillowWriter()

    N, S, M, conversations, mu_pos, mu_neg = good_animation_values(NSM=(2, 6, 6))

    fig, ax = plt.subplots(tight_layout=True)
    phi = initiate_phi(N, S, M)
    img = ax.matshow(phi[0], cmap='magma', vmin=0, vmax=1)
    cbar = fig.colorbar(img)
    ax.set_xlabel('meaning space')
    ax.set_ylabel('signal space')

    if method=='near_misses':
        anim = animation.FuncAnimation(fig, func_nearmisses, fargs=(img, phi, N, S, M, conversations, mu_pos, mu_neg, ), frames=50, interval=100)
        anim.save(f'animations/animation({N},{S},{M})_nearmisses.gif', writer=writergif)
    else:
        anim = animation.FuncAnimation(fig, func, fargs=(img, phi, N, S, M, conversations, mu_pos, mu_neg, ), frames=50, interval=100)
        anim.save(f'animations/animation({N},{S},{M}).gif', writer=writergif)

    if show_final:

        if N == 4:
            phi_final = phi
            fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(6,6))
            for i, ax in enumerate(axs.flat):
                ax.imshow(phi_final[i], cmap='magma', vmin=0, vmax=1)
                ax.axis('off')
                ax.set_title(f'speaker {i}')

            fig.savefig(f'animations/phi_final({N},{S},{M}).png')

run_animation(method='near_misses')

