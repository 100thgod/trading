
# Implements several machine learning algorithms

import numpy as np
from numpy.random import seed

# ----------------------- PERCEPTRON ALGORITHM

class linear_template(object):
    """ Template for ml algorithms using linear combinations of factors

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training dataset

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.

    """

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None

    def net_input(self, X):
        """ Calculate net input. """
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        """ Return class label after unit step. """
        return np.where(self.net_input(X) >= 0.0, 1, -1)



class Perceptron(linear_template):
    """ Perceptron classifier

    Additional Attributes
    ---------------------
    errors_ : list
        Number of misclassifications in evert epoch.

    """

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] = update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


#-------------------------- ADALINE ALGORITHM WITH GRADIENT DESCENT

class AdalineGD(linear_template):
    """ ADAptive LInear NEuron classifier with batch gradient descent.

    Additional Attributes
    ---------------------
    cost_ : list
        Cost function value in every epoch.

    """

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            errors = (y - self.net_input(X))
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self, X):
        """ Calculates net input. """
        return self.net_input(X)

    def predict(self, X):
        """ Return class label after unit step. """
        return np.where(self.activation(X) >= 0.0, 1, -1)


class AdalineSGD(linear_template):
    """ ADAptive LInear NEuron classifier with stochastic gradient descent

    Additional Attributes
    ---------------------
    cost_ : list
        Cost function value in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initialising the weights.

    """

    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        super(AdalineSGD, self).__init__(eta = eta, n_iter = n_iter)
        self.w_initialised = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.initialise_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """ Fit training data without reinitialising the weights. """
        if not self.w_initialised:
            self._initialise_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        ''' Shuffle training data. '''
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def initialise_weights(self, m):
        """ Initialise weights to zero. """
        self.w_ = np.zeros(1+m)
        self.w_initialised = True

    def _update_weights(self, xi, target):
        ''' Apply Adaline learning to update the weights. '''
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost



