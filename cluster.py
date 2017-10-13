import numpy as np
import os
import matplotlib.pyplot as plt

def euldist(points):
    len_pts=len(points)
    dist=np.zeros((len_pts,len_pts))
    for i in range(len_pts):
        for j in range(len_pts):
            dist[i,j]=np.sqrt(np.sum(np.square(points[i]-points[j])))
    dc=np.sort(np.concatenate(dist))[int(np.ceil(len_pts+0.02*len_pts*(len_pts-1)))]
    return dist,dc

def calrho(dist,dc):
    len_rho=len(dist)
    rho=np.zeros(len_rho)
    for i in range(len_rho):
        # rho[i]=np.sum(dist[i,:]<dc)
        rho[i]=np.sum(np.exp(-np.square(dist[i,:]/dc)))
    return rho

def caldelta(rho,dist):
    len_delta=len(rho)
    delta=np.ones(len_delta)*np.inf
    q=np.arange(len_delta)
    for i in range(len_delta):
        for j in range(len_delta):
            if (rho[j]>rho[i])&(dist[i,j]<delta[i]):
               delta[i]=dist[i,j]
               q[i]=j
    indexmax=np.argmax(delta)
    delta[indexmax]=dist[indexmax,:].max()
    return delta,q

def calcenters(gamma):
    x = np.flipud(np.argsort(gamma))
    y = np.flipud(np.sort(gamma))
    gamma_mean=gamma.mean()
    centers=[x[0],x[1]]
    for i in range(2, len(y) - 1):
        # if y[i] - y[i + 1] < (y[i - 1] - y[i]) / 2.:
        #     break
        if y[i]-gamma_mean<y[i-1]-y[i]:
            break
        centers.append(x[i])
    return centers

def calclusters(q,rho,centers):
    clusters=np.array(centers).reshape(-1,1).tolist()
    qc=np.copy(q)
    for i in np.flipud(np.argsort(rho)):
        if i not in centers:
            if qc[i] not in centers:
                qc[i]=qc[qc[i]]
                clusters[centers.index(qc[i])].append(i)
    return qc,clusters

def plot(rho,delta,gamma,points,clusters):
    # plt.figure(figsize=(12,18))
    # plt.subplot(221)
    # plt.scatter(rho,delta,color='k')
    # plt.xlabel(r'$\rho$')
    # plt.ylabel(r'$\delta$')
    # plt.subplot(222)
    # plt.scatter(np.arange(len(gamma)),np.sort(gamma,),color='r')
    # plt.ylabel('r$\gamma$')
    # plt.subplot(223)
    for cluster in clusters:
        plt.scatter(points[cluster][:,0],points[cluster][:,1],color=np.random.rand(3))

def run(points,plotclusters=True):
    dist,dc=euldist(points)
    rho=calrho(dist,dc)
    delta,q=caldelta(rho,dist)
    gamma=rho*delta
    centers=calcenters(gamma)
    qc,clusters=calclusters(q,rho,centers)
    if plotclusters:
        plot(rho,delta,gamma,points,clusters)
        plt.show()
    return clusters,centers

if __name__=='__main__':
    name_sets = 'D31.txt'
    # path_sets = os.path.join(os.path.expanduser('~'), 'DataSets/Cluster/Shape-sets', name_sets)
    path_sets=name_sets
    data = np.loadtxt(path_sets)
    points = data[:, 0:2]
    run(points)

