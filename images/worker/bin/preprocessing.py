#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
import pickle
import time

preprocessing_start = time.time()
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="", help='dataset path')
parser.add_argument('--label', type=str, default="", help='target label name')
parser.add_argument('--drop', type=str, default="", help='features to drop')
parser.add_argument('--random_state', type=int,
                    default=42, help='random state')
parser.add_argument('--test_size', type=float, default=0.2, help='test size')
parser.add_argument('--shuffle', default=False,
                    help='shuffle dataset', action='store_true')
parser.add_argument('--verbose', default=False,
                    help='verbose', action='store_true')
parser.add_argument('--scale', default=False,
                    help='scale dataset with standard scaler', action='store_true')
parser.add_argument('--mode', type=str, default="classification",help='classification or regression',choices=['classification','regression'])
parser.add_argument('--encode_label', default=False,help='encode target label', action='store_true')
# parser.add_argument('--output', type=str,
#                     default="preprocessing_output", help="output folder path")

args = parser.parse_args()

if args.dataset == "":
    print("Please specify dataset path")
    exit()

if args.label == "":
    print("Please specify target label name")
    exit()

dataset = pd.read_csv(args.dataset)
original_dataset = dataset.copy()

if args.verbose:
    print(dataset.head())
    for col in dataset.columns:
        print(f'Missing values in {col} : {dataset[col].isnull().sum()}')

# drop features if specified
if args.drop != "":
    features_to_drop = args.drop.split(",")
    for feature in features_to_drop:
        dataset = dataset.drop(feature, axis=1)

# divide dataset into features and target label
X = dataset.drop(args.label, axis=1)
y = dataset[args.label]

# get classes and features names
classes = np.unique(y)
features_names = X.columns

if args.encode_label:
    # encode target label
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
else:
    y = y.to_numpy()

# scale dataset with standard scaler if specified
if args.scale:
    scaler = StandardScaler()
    X = scaler.fit_transform(X.to_numpy())
else:
    X = X.to_numpy()

data_points = pd.DataFrame(X, columns=features_names)

# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, shuffle=args.shuffle)

# create output folder if not exists
# if not os.path.exists(args.output):
#     os.makedirs(args.output)

# logs preprocessing report if verbose is specified
if args.verbose:
    table = PrettyTable()
    table.field_names = ["Key", "Value", "Type"]
    table.add_row(
        ["Dataset Original", original_dataset.shape, type(original_dataset)])
    table.add_row(["Dataset Processed", dataset.shape, type(dataset)])
    table.add_row(["X_Train", X_train.shape, type(X_train)])
    table.add_row(["X_Test", X_test.shape, type(X_test)])
    table.add_row(["y_Train", y_train.shape, type(y_train)])
    table.add_row(["y_Test", y_test.shape, type(y_test)])
    table.add_row(["Dataset Scaled", args.scale, type(args.scale)])
    table.add_row(["Features", features_names.values, type(features_names)])
    table.add_row(["Target label", args.label, type(args.label)])
    table.add_row(["Classes", classes, type(classes)])
    print(table)

    print(data_points.describe())

    # filename = args.output + "/preprocessing_report.log"
    # with open(filename, 'w+') as f:
    #     f.write(str(table))
    #     f.write("\n\n")
    #     f.write(str(data_points.describe()))

# save preprocessed dataset

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
pickle.dump(features_names, open("feature_names.pkl", 'wb'))
pickle.dump(classes, open("classes.pkl", 'wb'))
if args.encode_label:
    pickle.dump(encoder, open("encoder.pkl", 'wb'))
if args.scale:
    pickle.dump(scaler, open("scaler.pkl", 'wb'))

preprocessing_end = time.time()


with open('preprocessing.txt', 'a+') as f:
    f.write(str(preprocessing_start)+" "+str(preprocessing_end)+" "+str(preprocessing_end - preprocessing_start)+"\n")
    f.close()