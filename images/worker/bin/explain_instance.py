#!/usr/bin/env python3

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
import time

explain_instance_start = time.time()


parser = argparse.ArgumentParser()

parser.add_argument('--instance', type=int, default=0, help='instance index')
parser.add_argument('--centroid_idx', type=str,
                    default="", help='centroid index path')

parser.add_argument('--X_train', type=str, default="", help='X_train path')
parser.add_argument('--X_test', type=str, default="", help='X_test path')
parser.add_argument('--y_train', type=str, default="", help='y_train path')
parser.add_argument('--y_test', type=str, default="", help='y_test path')

parser.add_argument('--model', type=str, default="", help='model path')
parser.add_argument('--random_state', type=int,
                    default=42, help='random state')

# Lime
parser.add_argument('--mode', type=str, default="classification",
                    help="classification or regression", choices=['classification', 'regression'])
parser.add_argument('--features', type=str, default="",
                    help="path to features file")
parser.add_argument('--classes', type=str, default="",
                    help="path to classes file")
parser.add_argument('--num_samples', type=int, default=1,
                    help="number of samples to generate")

parser.add_argument('--verbose', default=False,
                    help='verbose', action='store_true')
# parser.add_argument('--output', type=str,
#                     default="xai_output", help="output folder path")


args = parser.parse_args()

if args.model == "":
    print("Please specify model path")
    exit()

X_train = np.load(args.X_train)
X_test = np.load(args.X_test)
y_train = np.load(args.y_train)
y_test = np.load(args.y_test)
centroid_idx = pickle.load(open(args.centroid_idx,'rb'))

model = pickle.load(open(args.model, 'rb'))
features_names = pickle.load(open(args.features, 'rb'))
classes = pickle.load(open(args.classes, 'rb'))

# create output folder if not exists
# if not os.path.exists(args.output):
#     os.makedirs(args.output)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=features_names, class_names=classes, random_state=args.random_state, mode=args.mode)

explanation = explainer.explain_instance(X_train[args.instance], model.predict_proba, num_features=len(
        features_names), top_labels=len(classes), trainable=False, num_samples=args.num_samples,prefix_instance=str(centroid_idx)+"_",output_folder=str(Path(__file__).parent.resolve()))

pickle.dump(explanation,open("explanation.pkl",'wb'))

if args.verbose:
    print(explanation.as_list())
    print("Local Pred:",explanation.local_pred)
    print("Real Model Pred",model.predict_proba([X_train[args.instance]]))

explain_instance_end = time.time()

with open('/home/slot1_1/explain_instance.txt', 'a+') as f:
    f.write(str(explain_instance_end - explain_instance_start))
    f.close()