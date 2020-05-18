# import libraries
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
import pickle as pk
import seaborn as sb


# read dataset
def readData():
    data = np.array(pd.read_csv('newTao.csv'))
    test_data = np.array(pd.read_csv('newEl.csv'))
    return data, test_data


# elbow method to find optimal k for k means
def elbow(X):
    sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        print(k)
        k_means = KMeans(n_clusters=k, random_state=10)
        k_means.fit(X)
        sum_of_squared_distances.append(k_means.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('elbow method for optimal k')
    plt.show()


# find mean and covariance
def meanAndCov(clusters):
    perClusterMean = []
    perClusterCovariance = []
    for key in clusters:
        data = clusters[key]
        perClusterMean.append(np.mean(data, axis=0))
        perClusterCovariance.append(np.cov(data.T))
    return perClusterMean, perClusterCovariance


# multivariate probability distribution
def probDist(data, mean, cov):
    prob = []
    d = data.shape[1]
    term1 = np.power((2 * np.pi), (d/2))
    for row in data:
        probVals = []
        for myu, sig in zip(mean, cov):
            val = np.subtract(row, myu)
            mahabDist = -((1/2) * (np.dot(np.dot(val.T, np.linalg.inv(sig)), val)))
            probVals.append(((1 / (term1 * np.sqrt(np.linalg.det(sig)))) * np.exp(mahabDist)))
        prob.append(probVals)
    return prob


# k means algorithm
def kmeans(data, test_data, k):
    clusters = {}
    test_clusters = {}

    seed = 10
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)

    clust = kmeans.labels_
    split = int(0.7 * test_data.shape[0])
    pred = kmeans.predict(test_data[:split, :])
    clusterIds = list(set(clust))

    for id in clusterIds:
        clusters[id] = data[np.argwhere(clust == id).T[0], :]
        test_clusters[id] = test_data[np.argwhere(pred == id).T[0], :]

    mean, cov = meanAndCov(clusters)
    ranks = []
    for key in clusters:
        ranks.append(len(clusters[key]))

    ranks = np.flip(np.argsort(ranks).reshape((len(ranks), 1)))
    prob = np.array(probDist(test_data[split:, :], mean, cov))

    clustProb = np.empty((prob.shape[0], 1))
    for i in range(prob.shape[1]):
        clustProb = np.column_stack((clustProb, prob[:, i]/ np.sum(prob, axis=1)))
    clustProb = clustProb[:, 1:]
    plt.title("Test data cluster probability distribution")
    sb.heatmap(clustProb[:10, :], cbar_kws={'label': 'Probability Distribution'})
    plt.xlabel('Attributes')
    plt.ylabel('Samples')
    plt.show()
    print("Cluster {} determines high wind alert as maximum elnino data was clustered in it".format(ranks[0]))
    score = silhouette_score(test_data[:split, :], pred, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(k, score))


# dbscan algorithm
def dbScan(data, test_data):
    clusters = dict()
    test_clusters = dict()
    dbscan = DBSCAN(min_samples=10, eps=0.3).fit_predict(data)
    model = open('dbscanModel.model', 'wb')
    pk.dump(dbscan, model)
    model.close()
    model = open('dbscanModel.model', 'rb')
    dbscan = pk.load(model)
    split = int(0.7 * test_data.shape[0])
    clusterIds = list(set(dbscan))

    for id in clusterIds:
        clusters[id] = data[np.argwhere(dbscan == id).T[0], :]

    clusterCentres = []
    for id in clusters:
        clusterCentres.append([np.mean(clusters[id][:, 0]), np.mean(clusters[id][:, 1])])

    test = testData[:split, :]
    test = test[:, 7:]
    for coord in test:
        dist = []
        for centres in clusterCentres:
            dist.append(np.linalg.norm(coord - centres))
        id = np.argmin(dist)
        if id == len(clusterIds)-1:
            id = -1
        test_clusters.setdefault(id, []).append(coord)

    for cl in test_clusters:
        test_clusters[cl] = np.array(test_clusters[cl])

    pred = np.zeros(test.shape[0])
    for cl in test_clusters:
        points = test_clusters[cl]
        for point in points:
            x = np.argwhere(point[0] == test[:, 0])
            index = np.argwhere(point[1] == test[x, 1])[0]
            pred[index] = cl

    mean, cov = meanAndCov(clusters)
    ranks = []
    for key in clusters:
        ranks.append(len(clusters[key]))

    ranks = np.flip(np.argsort(ranks).reshape((len(ranks), 1)))
    prob = np.array(probDist(test_data[split:, :], mean, cov))
    clustProb = np.empty((prob.shape[0], 1))
    for i in range(prob.shape[1]):
        clustProb = np.column_stack((clustProb, prob[:, i] / np.sum(prob, axis=1)))
    clustProb = clustProb[:, 1:]
    plt.title("Test data cluster probability distribution")
    sb.heatmap(clustProb[:10, :], cbar_kws={'label': 'Probability Distribution'})
    plt.xlabel('Attributes')
    plt.ylabel('Samples')
    plt.show()
    print("Cluster {} determines high wind alert as maximum elnino data was clustered in it".format(ranks[0]))

    score = silhouette_score(test_data[:split, :], pred, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(len(clusterIds), score))


# main
if __name__ == '__main__':
    data, testData = readData()
    elbow(data[:, 7:])
    elbow(data[:, 10:])
    kmeans(data[:, 7:], testData[:, 4:], 4)
    kmeans(data[:, 10:], testData[:, 7:], 2)
    dbScan(data[:, 10:], testData[:, 7:])