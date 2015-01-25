"""
Assignment 1 for COMP652

Kelvin Xu
01/17/2014
"""

import numpy as np
import sklearn

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle


def load_data(loc_x='hw1x.dat', loc_y='hw1y.dat', normalize=False):
    """
    Part (a)
    Load data into X and y matrix
    """
    x, y = np.loadtxt(loc_x), np.loadtxt(loc_y)
    x, y = shuffle(x, y, random_state=1)
    if normalize:
        x /= np.max(x, axis=0)
    n_examples, n_feats = x.shape
    X = np.ones((n_examples, n_feats+1))
    X[:, :n_feats] = x
    return X, y


def polyRegress(x, d=2):
    """
    Part (d)
    Expand the x matrix with powers
    """
    if d < 2:
        return x
    (n_examples, n_feats) = x.shape
    n_feats -= 1
    xx = np.ones((n_examples, (n_feats)*d+1))
    for dd in xrange(0, d):
        xx[:, dd*n_feats:(dd+1)*n_feats] = x[:, :-1]**(dd+1)
    return xx


def Regress_gradient(x, y, x_val=None, y_val=None, alpha=0.0):
    """
    Implements gradient based optimization
    for least squares

    x - num examples by features 
    y - num examples x 1
    
    returns w: features x 1
    """ 
    w = numpy.randn(X.shape[1], 1)
    lr = 0.1
    lambda gradient w, x, y : np.dot(np.dot(x.T, x), w) - np.dot(x.T, y) + alpha * w   
    # TODO 

def Regress(X, y, alpha=0.0):
    """
    Ridge regression with sklearn,
    Alpha is the regularization parameter

    Assumes the input has been padded with a 1
    and normalized

    Same as:
    np.dot(np.linalg.inv(np.dot(x.T, x)-alpha * np.eye(x)), np.dot(x.T, y))
    """
    clf = Ridge(alpha=alpha, fit_intercept=False, tol=0.000001)
    clf.fit(X, y)
    return clf

def get_error(x, y, clf):
    y_pred = clf.predict(x)
    return ((y_pred - y)**2).sum() * 1. / y.shape[0]

def cross_valid_regress(x=None, y=None, alpha=0.):
    """
    This doesn't really make sense
    (d)
    """

    if x is None or y is None:
        x, y = load_data(normalize=True)
    # already shuffled on load
    kf = KFold(len(x), n_folds=5, shuffle=False)
    errors = np.zeros((5, 3, 5))
    for dd in range(1,6):
        for i, (train_val, test_idx) in enumerate(kf):
            # train, val, test
            dataset = [train_val[:-20], train_val[-20:], test_idx]
            # fit on train
            clf = Regress(polyRegress(x[dataset[0]], dd), y[dataset[0]])
            errors[dd-1, :, i] = np.array([get_error(polyRegress(x[idx], dd), y[idx], clf)
                                           for idx in dataset])
    # avereage across each iteration
    best_class = errors.mean(axis=2)[:,2].argmin() + 1
    best_hypothesis = Regress(polyRegress(x, best_class), y)
    return errors, best_hypothesis

def cross_valid_reg(x=None, y=None, alpha=0.):
    """
    (f)
    """
    if x is None or y is None:
        x, y = load_data(normalize=True)
    # already shuffled on load
    x = polyRegress(x, 4)
    kf = KFold(len(x), n_folds=5, shuffle=False)
    test_lambda = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 10]
    errors = np.zeros((len(test_lambda, 5)))
    for l_idx, l in enumerate(test_lambda):
        for ii, (train_idx, test_idx) in enumerate(kf): 
            clf = Regression(x[train_idx], y[train_indx])
            
            errors[l_idx, ii] = np.array([get_error(x[idx], y[idx], clf) 
                                          for idx in [train_idx, test_idx]])
    return errors
