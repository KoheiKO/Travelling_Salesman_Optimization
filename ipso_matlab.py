import numpy as np
import matplotlib.pyplot as plt

def optimize(x, city_coor, city_dist):
    m = x.shape[0]
    n = city_coor.shape[0]
    Optim = np.zeros(m)

    for i in range(m):
        for j in range(n - 1):
            Optim[i] += city_dist[x[i, j] - 1, x[i, j + 1] - 1]

        Optim[i] += city_dist[x[i, 0] - 1, x[i, n - 1] - 1]

    return Optim

# Load the data from eil51.txt or your data source
data2 = np.loadtxt('eil51.txt')
paddpoints = data2[:, :2]
n = paddpoints.shape[0]
t = np.arange(1, n + 1)
paddpoints = np.column_stack((paddpoints, t))

theta = np.linspace(0, 2 * np.pi, 100)

indiNumber = 1000
nMax = 100

# Initialize cityDist and individual
cityDist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            cityDist[i, j] = np.sqrt((paddpoints[i, 0] - paddpoints[j, 0]) ** 2 +
                                   (paddpoints[i, 1] - paddpoints[j, 1]) ** 2)
        cityDist[j, i] = cityDist[i, j]

individual = np.zeros((indiNumber, n), dtype=int)

# Initialize each particle with a random route
for i in range(indiNumber):
    individual[i, :] = np.random.permutation(n) + 1

Optim = optimize(individual, paddpoints, cityDist)
index = np.argmin(Optim)
indiFit1 = [Optim[index], index]
tourPbest = individual.copy()
tourGbest = individual[index, :].copy()
recordPbest = np.full(indiNumber, np.inf)
recordGbest = Optim[index]
xnew1 = individual.copy()

c1 = 0.7
c2 = 0.1

L_best = np.zeros(nMax)

for N in range(nMax):
    Optim = optimize(individual, paddpoints, cityDist)

    for i in range(indiNumber):
        if Optim[i] < recordPbest[i]:
            recordPbest[i] = Optim[i]
            tourPbest[i, :] = individual[i, :]
        if Optim[i] < recordGbest:
            recordGbest = Optim[i]
            tourGbest = individual[i, :]

    index = np.argmin(recordPbest)
    recordGbest = recordPbest[index]

    for i in range(indiNumber):
        r1 = np.round(c1 * np.random.rand() * (n + 1))
        while r1 < 4:
            r1 = np.round(c1 * np.random.rand() * (n + 1))
        r2 = np.round(c2 * np.random.rand() * (n + 1))
        while r2 < 2:
            r2 = np.round(c2 * np.random.rand() * (n + 1))

        sr1 = np.random.randint(n)
        sr2 = np.random.randint(n)

        pd = tourPbest[i, sr1:sr1 + r1] if sr1 + r1 - 1 <= n else np.concatenate((tourPbest[i, sr1:n], tourPbest[i, :sr1 + r1 - 1 - n]), axis=None)
        ld = tourGbest[sr2:sr2 + r2] if sr2 + r2 - 1 <= n else np.concatenate((tourGbest[sr2:], tourGbest[:sr2 + r2 - 1 - n]), axis=None)

        for ip in range(r1):
            for il in range(r2):
                if pd[ip] == ld[il]:
                    pd[ip] = 0
        pd = pd[pd != 0]
        pdd = pd.copy()
        szpdd = pdd.size
        for ix in range(n):
            for il in range(r2):
                if individual[i, ix] == ld[il]:
                    individual[i, ix] = 0
            for ipdd in range(szpdd):
                if individual[i, ix] == pdd[ipdd]:
                    individual[i, ix] = 0
        pi1 = individual[i, :]
        ind = pi1 == 0
        pi = pi1[~ind]

        if szpdd > 1:
            pifirst = pdd[0]
            piend = pdd[-1]
            szpi = pi.size
            pid = np.zeros((szpi, szpi + 2), dtype=int)
            pidfit = np.zeros((szpi, n - 1))
            pidfit1 = np.zeros(szpi)

            for ipi in range(szpi):
                pid[ipi, :] = np.concatenate((pi[:ipi], pifirst, pi[ipi + 1:]), axis=None)

                for x in range(szpi + 1):
                    pidfit[ipi, x] = np.sqrt((paddpoints[pid[ipi, x] - 1, 0] - paddpoints[pid[ipi, x + 1] - 1, 0]) ** 2 +
                                           (paddpoints[pid[ipi, x] - 1, 1] - paddpoints[pid[ipi, x + 1] - 1, 1]) ** 2)

                pidfit1[ipi] = np.sum(pidfit[ipi, :])
                pidfit1[ipi] = pidfit1[ipi] + np.sqrt((paddpoints[pid[ipi, 0] - 1, 0] - paddpoints[pid[ipi, -1] - 1, 0]) ** 2 +
                                                   (paddpoints[pid[ipi, 0] - 1, 1] - paddpoints[pid[ipi, -1] - 1, 1]) ** 2)

            idx = np.argmin(pidfit1)
            xd1 = pid[idx, :]

            if idx == szpi + 1:
                xd = np.concatenate((xd1[:-2], pdd), axis=None)
            else:
                xd = np.concatenate((xd1[:idx], pdd, xd1[idx + 3:]), axis=None)
            szxd = xd.size
            ldfirst = ld[0]
            ldend = ld[-1]
            xdd = np.zeros((szxd, szxd + 2), dtype=int)
            xddfit = np.zeros((szxd, n - 1))
            xddfit1 = np.zeros(szxd)
            xdd1 =
            for xdi in range(szxd):
                xdd[xdi, :] = np.concatenate((xd[:xdi], ldfirst, ldend, xd[xdi + 1:]), axis=None)

                for x in range(szxd + 1):
                    xddfit[xdi, x] = np.sqrt((paddpoints[xdd[xdi, x] - 1, 0] - paddpoints[xdd[xdi, x + 1] - 1, 0]) ** 2 +
                                           (paddpoints[xdd[xdi, x] - 1, 1] - paddpoints[xdd[xdi, x + 1] - 1, 1]) ** 2)

                xddfit1[xdi] = np.sum(xddfit[xdi, :])
                xddfit1[xdi] = xddfit1[xdi] + np.sqrt((paddpoints[xdd[xdi, 0] - 1, 0] - paddpoints[xdd[xdi, -1] - 1, 0]) ** 2 +
                                                   (paddpoints[xdd[xdi, 0] - 1, 1] - paddpoints[xdd[xdi, -1] - 1, 1]) ** 2)

            idx2 = np.argmin(xddfit1)
            xdd1 = xdd[idx2, :]

            if idx2 == szxd:
                xdd1 = np.concatenate((xdd1[:-2], ld), axis=None)
            else:
                xdd1 = np.concatenate((xdd1[:idx2], ld, xdd1[idx2 + 3:]), axis=None)
            individual[i, :] = xdd1

    value, index = np.min(Optim)
    indiFit2 = [value, index]
    L_best[N] = Optim[index]

# Reorder the Gbest to start from city 1
gb = np.where(tourGbest == 1)
if gb != 0:
    tourGbest = np.concatenate((tourGbest[gb:], tourGbest[:gb]), axis=None)

Gbestcourse = tourGbest
minbest = np.min(L_best)

np.savetxt('Gbestcourse.txt', Gbestcourse, fmt='%d')

# Visualization
plt.figure()
plt.plot([paddpoints[Gbestcourse[0] - 1, 0], paddpoints[Gbestcourse[-1] - 1, 0]],
         [paddpoints[Gbestcourse[0] - 1, 1], paddpoints[Gbestcourse[-1] - 1, 1]], 'bo-', linewidth=1)

cx, cy = 0, 0
r = 25
plt.plot(r * np.sin(theta) + cx, r * np.cos(theta) + cy, 'r-', linewidth=0.5)

for i in range(1, n):
    plt.plot([paddpoints[Gbestcourse[i - 1] - 1, 0], paddpoints[Gbestcourse[i] - 1, 0]],
             [paddpoints[Gbestcourse[i - 1] - 1, 1], paddpoints[Gbestcourse[i] - 1, 1]], 'b-', linewidth=1)

plt.scatter(paddpoints[:, 0], paddpoints[:, 1], color='blue', marker='o')
plt.legend(['Proposed path', 'Survey area'], fontsize=16)
plt.grid(True)
plt.box(True)
plt.axis('equal')
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.xlabel('x-axis [m]', fontsize=16)
plt.ylabel('y-axis [m]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
