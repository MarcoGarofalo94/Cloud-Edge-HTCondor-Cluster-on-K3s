#!/usr/bin/env python3

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import argparse
import pickle
from pathlib import Path

#sleep in python
import time

#start time
linear_models_start = time.time()


parser = argparse.ArgumentParser()
print(str(Path(__file__).parent.resolve()))
#Dataset
parser.add_argument('--X_train', type=str, default="", help='X_train path')
parser.add_argument('--X_test', type=str, default="", help='X_test path')
parser.add_argument('--y_train', type=str, default="", help='y_train path')
parser.add_argument('--y_test', type=str, default="", help='y_test path')

#Model and centroids
parser.add_argument('--model', type=str, default="", help='model path')
parser.add_argument('--random_state', type=int,
                    default=42, help='random state')
parser.add_argument('--centroids', type=str, default="", help='centroids path')
parser.add_argument('--centroid_idx', type=int, default=0, help='centroid index')
parser.add_argument('--scaler', type=str, default="", help="scaler path")

#Lime
parser.add_argument('--mode', type=str, default="classification",
                    help="classification or regression", choices=['classification', 'regression'])
parser.add_argument('--features', type=str, default="",
                    help="path to features file")
parser.add_argument('--classes', type=str, default="",
                    help="path to classes file")
parser.add_argument('--num_samples', type=int, default=5000,
                    help="number of samples to generate")

parser.add_argument('--verbose', default=False,
                    help='verbose', action='store_true')
# parser.add_argument('--output', type=str,
#                     default="xai_output", help="output folder path")

args = parser.parse_args()

if args.model == "":
    print("Please specify model path")
    exit()
if args.centroids == "":
    print("Please specify centroids path")
    exit()

X_train = np.load(args.X_train)
X_test = np.load(args.X_test)
y_train = np.load(args.y_train)
y_test = np.load(args.y_test)

centroids = np.load(args.centroids)

model = pickle.load(open(args.model, 'rb'))
features_names = pickle.load(open(args.features, 'rb'))
classes = pickle.load(open(args.classes, 'rb'))

if args.scaler != "":
    scaler = pickle.load(open(args.scaler, 'rb'))
    X_train = scaler.inverse_transform(X_train)
    X_test = scaler.inverse_transform(X_test)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=features_names, random_state=args.random_state, class_names=classes, verbose=args.verbose, mode=args.mode)
# print(explainer.explain_instance(test[0], clf.predict_proba, num_features=2).as_list())

explainer.explain_instance(centroids[args.centroid_idx], model.predict_proba, num_features=len(
        features_names), top_labels=len(classes), trainable=True, num_samples=args.num_samples,prefix_instance=str(args.centroid_idx)+"_",output_folder=str(Path(__file__).parent.resolve()))
linear_models_end = time.time()

with open('/home/times/linear_models.txt', 'a+') as f:
    f.write(str(linear_models_end - linear_models_start)+"\n")
    f.close()