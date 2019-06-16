#! /usr/bin/env python3
from keras.datasets import mnist
from sklearn import svm
import numpy as np
import random

LABELS = [0, 1, 2, 3, 4]
TRAIN_SIZE = 500

def filter_data(x_train, y_train, x_test, y_test):
    def flat_normalize(image):
        return [x/255 for x in image.flatten()]

    X_train = []
    for label in LABELS:
        X_train += list(x_train[y_train == label])[:TRAIN_SIZE]
    X_train = np.array([flat_normalize(x) for x in X_train])    
    
    Y_train = np.array([label for label in LABELS for _ in range(TRAIN_SIZE)])
    print('train set: OK')
    
    print(X_train.shape)
    print(Y_train.shape)

    X_test = x_test[np.isin(y_test, LABELS)]
    X_test = np.array([flat_normalize(x) for x in X_test])

    Y_test = y_test[np.isin(y_test, LABELS)]
    print('test set: OK')

    train_set = list(zip(X_train, Y_train))
    random.shuffle(train_set)
    X_train, Y_train = zip(*train_set)
    print('shuffle train set: OK')
    return (X_train, Y_train), (X_test, Y_test)


def accuracy(clf, X_test, Y_test):
    return sum(clf.predict(X_test) == Y_test) / len(Y_test)

def k_folds(A, k=5):
    n = len(A)
    fold_size = int(np.ceil(n / k))
    folds = [A[i : i + fold_size] for i in range(0, n, fold_size)]
    train_folds = [
        np.concatenate([f1 for f1 in folds if not np.array_equal(f1, f2)]) for f2 in folds]

    return list(zip(train_folds, folds))

def cross_validation(X_folds, Y_folds, classifier):
    accuracy_list = []
    for X_fold, Y_fold in zip(X_folds, Y_folds):
        X_train, X_test = X_fold
        Y_train, Y_test = Y_fold

        clf = classifier
        clf.fit(X_train, Y_train)
        accuracy_list += [accuracy(clf, X_test, Y_test)]
    
    return sum(accuracy_list) / len(accuracy_list)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('loading data: OK')

    (X_train, Y_train), (X_test, Y_test) = filter_data(
        x_train, y_train, x_test, y_test)

    GAMMA = 0.05
    
    clf = svm.SVC(gamma=GAMMA)
    clf.fit(X_train, Y_train)
    print(accuracy(clf, X_test, Y_test))

    X_folds, Y_folds = k_folds(X_train), k_folds(Y_train)

    cross_validation_accuracy = cross_validation(
        X_folds, Y_folds, svm.SVC(gamma=GAMMA))
    print(cross_validation_accuracy)
    

