#!/usr/bin/env python3

import argparse
import numpy as np
import pickle
import time

similarity_start = time.time()

parser = argparse.ArgumentParser()

parser.add_argument('--alg', type=str, default="", help='clucstering algorithm',
                    choices=['kmeans', 'dbscan', 'affinity'])
parser.add_argument('--centroids', type=str, default="", help='centroid path')


parser.add_argument('--X_train', type=str, default="", help='X_train path')
parser.add_argument('--X_test', type=str, default="", help='X_test path')
parser.add_argument('--instance', type=int, default=0, help='instance index')
parser.add_argument('--use_test_set', default=False,
                    help='use test set', action='store_true')

parser.add_argument('--verbose', default=False,
                    help='verbose', action='store_true')
# parser.add_argument('--output', type=str,
#                     default="similarity_output", help="output folder path")
parser.add_argument('--dbscan_metric', type=str, default="euclidean", help='metric for DBSCAN',
                    choices=['euclidean', 'manhattan', 'cosine', 'l1', 'l2'])
# parser.add_argument('--hdbscan_metric', type=str, default="euclidean", help='metric for HDBSCAN',
#                     choices=['euclidean', 'manhattan', 'cosine', 'l1', 'l2' ])

args = parser.parse_args()


if args.alg == "":
    print("Please specify a valid algorithm")
    exit()

if args.centroids == "":
    print("Please specify a valid centroid path")
    exit()

centroids = np.load(args.centroids)
X_train = np.load(args.X_train)
instance = X_train[args.instance]

if args.use_test_set:
    X_test = np.load(args.X_test)
    instance = X_test[args.instance]

most_similar_centroid = np.ones(shape=centroids.shape) * np.inf
most_similar_centroid_idx = np.inf
distance_from_centroid = np.inf
if args.alg == "kmeans" or args.alg == "affinity":
    for idx, centroid in enumerate(centroids):
        distance_from_centroid = np.linalg.norm(centroid - instance)
        if distance_from_centroid < np.linalg.norm(most_similar_centroid - instance):
            most_similar_centroid = centroid
            most_similar_centroid_idx = idx

elif args.alg == "dbscan":
    # euclidean distance = sqrt(sum((x - y)^2)), 0 = most similar, inf = most dissimilar
    if args.dbscan_metric == "euclidean" or args.dbscan_metric == "l2":
        for idx, centroid in enumerate(centroids):
            distance_from_centroid = np.linalg.norm(centroid - instance)
            if distance_from_centroid < np.linalg.norm(most_similar_centroid - instance):
                most_similar_centroid = centroid
                most_similar_centroid_idx = idx
    # manhattan distance = sum of absolute differences between points, 0 = most similar, inf = most dissimilar
    if args.dbscan_metric == "manhattan" or args.dbscan_metric == "l1":
        for idx, centroid in enumerate(centroids):
            distance_from_centroid = np.sum(np.abs(centroid - instance))
            if distance_from_centroid < np.sum(np.abs(most_similar_centroid - instance)):
                most_similar_centroid = centroid
                most_similar_centroid_idx = idx
    # cosine similarity = dot product / (norm(a) * norm(b)), 1 = most similar, -1 = most dissimilar
    if args.dbscan_metric == "cosine":
        most_similar_centroid = centroids[0]
        for idx, centroid in enumerate(centroids):
            distance_from_centroid = 1 - \
                (np.dot(centroid, instance) /
                 (np.linalg.norm(centroid) * np.linalg.norm(instance)))
            if 1 - distance_from_centroid >= np.dot(most_similar_centroid, instance) / (np.linalg.norm(most_similar_centroid) * np.linalg.norm(instance)):
                most_similar_centroid = centroid
                most_similar_centroid_idx = idx

if args.verbose:
    print("Similarity metric based on: ", args.alg)
    print("Length Centroids: ", len(centroids))
    print("Centroids: \n", centroids)
    print("Instance to explain: ", instance)
    print("Most Similar Centroid: ", most_similar_centroid)
    print("Most Similar Centroid Index: ", most_similar_centroid_idx)
    print("Distance: ", distance_from_centroid)


# if not os.path.exists(args.output):
#     os.makedirs(args.output)
np.save("most_similar_centroid.npy", most_similar_centroid)
pickle.dump(most_similar_centroid_idx, open(
    "most_similar_centroid_idx.pkl", 'wb'))
similarity_end = time.time()

with open('/home/times/similarity.txt', 'a+') as f:
    f.write(str(similarity_end - similarity_start)+"\n")
    f.close()