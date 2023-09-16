#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import argparse
import os
import time

svm_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--X_train', type=str, default="", help='X_train path')
parser.add_argument('--X_test', type=str, default="", help='X_test path')
parser.add_argument('--y_train', type=str, default="", help='y_train path')
parser.add_argument('--y_test', type=str, default="", help='y_test path')
parser.add_argument('--random_state', type=int, default=42, help='random state')
parser.add_argument('--verbose', default=False, help='verbose', action='store_true')
parser.add_argument('--mode',type=str,default="classification",help='classification or regression',choices=['classification','regression'])

# parser.add_argument('--output', type=str, default="models_output", help='output file')
args = parser.parse_args()

if args.X_train == "" or args.X_test == "" or args.y_train == "" or args.y_test == "":
    print("Please specify X_train, X_test, y_train and y_test paths")
    exit()

X_train = np.load(args.X_train)
X_test = np.load(args.X_test)
y_train = np.load(args.y_train)
y_test = np.load(args.y_test)

parameters = {'C':[0.001,0.01,0.1,1,10,100], 'kernel':['rbf','linear', 'poly', 'sigmoid'], 'gamma': ['scale','auto']}
svm = SVC(C='C', kernel='kernel', gamma='gamma',probability=True,random_state=args.random_state)
# print('Searching for best parameters...')
searcher = GridSearchCV(estimator=svm, param_grid=parameters, cv=2,verbose=args.verbose)
searcher.fit(X_train, y_train)
# print('Best parameters', str(searcher.best_params_))
# print('Best estimators', str(searcher.best_estimator_))
# print('Best estimators', '{:.2%}'.format(searcher.best_score_))
clf_svm = SVC(C=searcher.best_params_['C'], kernel=searcher.best_params_['kernel'], gamma=searcher.best_params_['gamma'],probability=True,random_state=args.random_state)
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
# print('Error rate:', str(np.sum(y_test != clf_svm.predict(X_test))))

# create output folder if not exists
# if not os.path.exists(args.output):
#     os.makedirs(args.output)
pickle.dump(clf_svm, open('svm.pkl', 'wb'))
svm_end = time.time()

with open('svm.txt', 'a+') as f:
    f.write(str(svm_start)+" "+str(svm_end)+" "+str(svm_end - svm_start)+"\n")
    f.close()