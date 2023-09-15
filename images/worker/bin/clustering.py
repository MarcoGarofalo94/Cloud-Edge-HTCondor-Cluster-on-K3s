#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AffinityPropagation
from kneed import KneeLocator
import argparse
import time

clustering_start = time.time()
 

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', default=False, help='verbose', action='store_true')
parser.add_argument('--X_train', type=str, default="", help='X_train path')
parser.add_argument('--X_test', type=str, default="", help='X_test path')
parser.add_argument('--y_train', type=str, default="", help='y_train path')
parser.add_argument('--y_test', type=str, default="", help='y_test path')
parser.add_argument('--use_all_data', default=False, help='use all dataset', action='store_true')
parser.add_argument('--alg', type=str, default="", help='clucstering algorithm',choices=['kmeans', 'dbscan', 'affinity'])
parser.add_argument('--max_features', type=int, default=2, help='number of features to use')
parser.add_argument('--dbscan_eps', type=float, default=0.3, help='epsilon for DBSCAN')
parser.add_argument('--dbscan_min_samples', type=int, default=5, help='min_samples for DBSCAN')
parser.add_argument('--dbscan_metric', type=str, default="euclidean", help='metric for DBSCAN',choices=['euclidean', 'manhattan', 'cosine', 'l1', 'l2',])
parser.add_argument('--random_state', type=int, default=42, help='n_clusters for KMeans')
parser.add_argument('--kmeans_n_clusters', type=int, default=None, help='n_clusters for KMeans')
parser.add_argument('--kmeans_max_clusters', type=int, default=11, help='number of clusters')
parser.add_argument('--kneedle_curve', type=str, default="convex", help='curve type for kneedle',choices=['convex', 'concave'])
parser.add_argument('--kneedle_direction', type=str, default="decreasing", help='direction for kneedle',choices=['decreasing', 'increasing'])
parser.add_argument('--kneedle_s', type=float, default=1.0, help='S for kneedle')
parser.add_argument('--kmeans_max_iter', type=int, default=300, help='Maximum iterarions for KMeans',)
parser.add_argument('--kmeans_init', type=str, default="k-means++", help='init for KMeans',choices=['k-means++', 'random'])
parser.add_argument('--kmeans_n_init', type=int, default=10, help='n_init for KMeans')
# parser.add_argument('--output', type=str, default="clustering_output", help='output file')
args = parser.parse_args()

if args.X_train == "" or args.X_test == "" or args.y_train == "" or args.y_test == "":
    print("Please specify X_train, X_test, y_train and y_test paths")
    exit()

X_train = np.load(args.X_train)
X_test = np.load(args.X_test)
y_train = np.load(args.y_train)
y_test = np.load(args.y_test)

if args.use_all_data:
    X_train = np.concatenate((X_train, X_test))
    y_train = np.concatenate((y_train, y_test))
    
def estimate_centroids(X, y, clusters):
    centroids = np.array([]).reshape(0, X.shape[1])
    for cluster in clusters:
        if cluster != -1:
          row_ix = np.where(y == cluster)
          estimated_centroid =np.array(np.mean(X[row_ix]))
          most_similar = np.ones(shape=estimated_centroid.shape) * np.inf
          for row in X[row_ix]:
            if np.linalg.norm(row - estimated_centroid) < np.linalg.norm(most_similar - estimated_centroid):
              most_similar = row
          centroids = np.append(centroids, most_similar)

        # else:
        #   row_ix = np.where(y == cluster)
        #   centroids = np.concatenate((centroids,X[row_ix]))
    return np.array(centroids).reshape(-1, X.shape[1])


if args.alg == "kmeans":
    # if no n_clusters is specified, use kneedle to find the best n_clusters
    if args.kmeans_n_clusters is None: 
      wcss = []
      for i in range(1, args.kmeans_max_clusters):
          kmeans = KMeans(n_clusters = i, init = args.kmeans_init, max_iter = args.kmeans_max_iter, n_init = args.kmeans_n_init, random_state = args.random_state)
          kmeans.fit(X_train)
          wcss.append(kmeans.inertia_)
      kneedle = KneeLocator(range(1, args.kmeans_max_clusters), wcss, S=args.kneedle_s, curve=args.kneedle_curve, direction=args.kneedle_direction)
      n_clusters = kneedle.knee
    else:
      n_clusters = args.kmeans_n_clusters
    
    kmeans = KMeans(n_clusters = n_clusters, init = args.kmeans_init, max_iter = args.kmeans_max_iter, n_init = args.kmeans_n_init, random_state = args.random_state)
    y_kmeans = kmeans.fit_predict(X_train)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

elif args.alg == "dbscan":
    dbscan = DBSCAN(eps=args.dbscan_eps,min_samples=args.dbscan_min_samples,metric=args.dbscan_metric)
    y_pred = dbscan.fit_predict(X_train)
    labels = dbscan.labels_
    centroids = estimate_centroids(X_train, y_pred, np.unique(y_pred))

elif args.alg == "affinity":
    af = AffinityPropagation(random_state=args.random_state).fit(X_train)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    centroids = af.cluster_centers_
else:
    print("Please specify clustering algorithm (kmeans,dbscan,affinity)")
    exit()

#save centroids

# create output folder if not exists
# if not os.path.exists(args.output):
#     os.makedirs(args.output)

if args.verbose:
    print("Centroids:")
    print(centroids)
    print("Labels:")
    print(labels)

    print(type(centroids),centroids.shape)

np.save("centroids.npy", centroids)

clustering_end = time.time()

with open('/home/times/clustering.txt', 'a+') as f:
    f.write(str(clustering_end - clustering_start)+"\n")
    f.close()