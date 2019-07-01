import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig, svd

df = pd.read_csv('./pm2_5.txt')

#sa = np.arange(4000) / 4000
sa = df['PM_City Station'].values
sb = df['PM_Jingan'].values

def outMul(vectorA, vectorB):
    return vectorA.reshape(-1, 1).dot(vectorB.reshape(-1, 1).T)

k = 1
w = 30
m = 10
beta = 0.9

outMat_a = []
outMat_b = []
for i in range(len(sa) - w):
    outMat_a.append(outMul(sa[i:i+w], sa[i:i+w]))
    outMat_b.append(outMul(sb[i:i+w], sb[i:i+w]))

expMat_a = [outMat_a[0]]
expMat_b = [outMat_b[0]]
for i in range(1, len(outMat_a)):
    scale = (1 - beta) / (1 - beta ** (1+i))
    expMat_a.append((beta * expMat_a[-1] + outMat_a[i]) * scale)
    
    expMat_b.append((beta * expMat_b[-1] + outMat_b[i]) * scale)

def loco(ux, uy, wa, wb):
    first = np.linalg.norm(ux.dot(wb.reshape(-1, 1)))
    second = np.linalg.norm(uy.dot(wa.reshape(-1, 1)))
    return 0.5 * (first + second)

score = []
ks = []
for ma, mb in zip(expMat_a, expMat_b):
    ua, va, _ = svd(ma)
    ub, vb, _ = svd(mb)

    cumA = np.cumsum(va) / np.sum(va)
    k = 1
    while cumA[k] < 0.98:
        k += 1
    ks.append(k)
    k = 4
    ux = ua[:, :k].T
    uy = ub[:, :k].T
    
    score.append(loco(ux, uy, ua[:, 0], ub[:, 0]))

k = sum(ks) / len(ks)

fig, axes = plt.subplots(2, 1)
fig.subplots_adjust(hspace = 0.5)
axes[0].plot(range(len(sa)), sa, range(len(sb)), sb)
axes[1].plot(score, lw = 0.6)
axes[1].set_title('k = {}'.format(k))

plt.show()