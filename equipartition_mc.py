import numpy as np
import matplotlib.pyplot as plt
import sys

def initiate_values():
    if len(sys.argv) != 4:
        S, M, n = 2, 8, 100000
    else:
        S = int(sys.argv[1])
        M = int(sys.argv[2])
        n = int(sys.argv[3])
    
    return S, M, n

S, M, n = initiate_values()

dic = {}

for i in range(n):

    arr = np.zeros((S,M))
    for meaning in range(M):
        signal = np.random.randint(S) # one signal per meaning
        arr[signal, meaning] = 1

    # indices = np.random.choice(S*M - 1, size=M, replace=False)
    # arr[indices] = 1
    # arr = arr.reshape((S,M))

    sm_distr = arr.sum(axis=1).astype(int)
    sm_distr = np.sort(sm_distr)
    string = ''.join(str(x) for x in sm_distr)

    if string in dic:
        dic[string] += 1
    else:
        dic[string] = 1

def factorial(n):
    return 1 if (n==1 or n==0) else n * factorial(n - 1)

def multiplicity(key_str):

    k_list = [int(x) for x in key_str]
    S = len(k_list)
    M = sum(k_list)
    k_set = list(set(k_list))
    denom1 = 1
    for k in k_list: denom1 *= factorial(k)
    denom2 = 1
    for k_ in k_set:
        n_k = k_list.count(k_)
        denom2 *= factorial(n_k) # if any blocks of same length, take factorial

        # i don't really understand why this step is required (taking it out produces worse results)

    # total number of possible arrangements S**M (S choices for the M meanings)

    return S**(-M) * factorial(M) / denom1 * factorial(S) / denom2

def get_multiplicities(keys_list):

    multiplicities = []
    for key in keys_list:
        multiplicities.append(multiplicity(key))
    
    return np.array(multiplicities)


fig, ax = plt.subplots(tight_layout=True)
x, y = np.array(list(dic.keys())), np.array(list(dic.values()))
y = y / y.sum() # normalised
inds = np.argsort(-y)

mult = get_multiplicities(list(dic.keys()))

ax.bar(x[inds], y[inds], color='teal', alpha=0.5, label='Monte Carlo simulation') #100/n np.array()
# ax.plot(x[inds], y[inds], 'k--')
ax.plot(x[inds], mult[inds], 'k--',label='theoretical multiplicity')
ax.set_title(f'$(S, M) = ({S}, {M})$ - {n} runs')
ax.legend()
plt.show()

        

    
