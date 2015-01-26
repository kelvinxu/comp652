"""
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
    

def get_error(X, y, w):
    if isinstance(w, sklearn.linear_model.ridge.Ridge):
        y_pred = w.predict(X)
    else:
        y_pred = np.dot(X,w)
    return ((y_pred - y)**2).sum() * 1. / y.shape[0]
    

def Regress_exact(X, y, alpha=0.0):
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


def Regress_gradient(x, y, x_val, y_val, max_iter=10000, alpha=0.0, patience=20):
    """
    Implements gradient based optimization
    for least squares

    x - num examples by features
    y - num examples x 1
    patience - the number of updates to weight to see if the loss goes down 

    returns w: features x 1
    """
    w = np.random.randn(x.shape[1],)
    lr = 0.0001

    gradient = lambda w, x, y : np.dot(np.dot(x.T, x), w) - np.dot(x.T, y).T + alpha * w

    history_err = []
    w_delta = 0

    for i in range(0, max_iter):
        w_old = w
        dw = gradient(w, x, y)
        w += -lr * dw + 0.9 * w_delta
        w_delta = w - w_old
        train_error = get_error(x, y, w)
        val_error = get_error(x_val, y_val, w)
        history_err.append([train_error, val_error])
        if i == 0 or val_error < np.array(history_err)[:,1].min():
            best_w = w
            bad_counter = 0
        if i > patience and val_error >= np.array(history_err)[:-patience,1].min():
            bad_counter += 1
            if bad_counter > patience:
                break
    return best_w, get_error(x, y, w), get_error(x_val, y_val, w)


def cross_valid_regress(x=None, y=None, alpha=0., d=5, n_folds=5):
    """
    (d)
    """
    if x is None or y is None:
        x, y = load_data(normalize=True)
    # already shuffled on load
    kf = KFold(len(x), n_folds=n_folds, shuffle=False)
    # error is #degrees by # entries by # folds
    errors = np.zeros((n_folds, 3, d))
    for dd in range(1,d+1):
        for i, (train_val, test_idx) in enumerate(kf):
            # train, val, test
            dataset = [train_val[:-20], train_val[-20:], test_idx]
            # fit on train
            w, train_err, val_err = Regress_gradient(polyRegress(x[dataset[0]], dd), y[dataset[0]],
                                                       polyRegress(x[dataset[1]], dd), y[dataset[1]])
            # compute test
            test_err = get_error(polyRegress(x[dataset[2]], dd), y[dataset[2]], w)
            errors[dd-1, :, i] = np.array([train_err, val_err, test_err])
        print "Tried d = %d.." % dd
    # average across each fold
    best_class = errors.mean(axis=2)[:,2].argmin() + 1
    best_hypothesis = Regress_exact(polyRegress(x, best_class), y)
    return errors, best_class, best_hypothesis


def cross_valid_regularize(x=None, y=None, exact=False):
    """
    (f)
    """
    if x is None or y is None:
        x, y = load_data(normalize=True)
    # already shuffled on load
    x = polyRegress(x, 4)
    kf = KFold(len(x), n_folds=5, shuffle=False)
    test_lambda = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 10]
    errors = np.zeros((len(test_lambda), 2, 5))
    for l_idx, l in enumerate(test_lambda):
        for ii, (train_idx, test_idx) in enumerate(kf):
            if exact:
                w = Regress_exact(x[train_idx], y[train_idx], alpha=l)
            else:
                w = Regress_gradient(x[train_idx], y[train_idx], x[test_idx], y[test_idx], alpha=l)
                w = w[0]
            errors[l_idx,:, ii] = np.array([get_error(x[idx], y[idx], w)
                                          for idx in [train_idx, test_idx]])
    return errors
