
"""

Good references:

- Feature selection: http://scikit-learn.org/stable/modules/feature_selection.html

- Nonlinear dimension reduction: http://scikit-learn.org/stable/modules/manifold.html

- Different scoring parameters: http://scikit-learn.org/stable/modules/model_evaluation.html

- Precision-recall curve: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

"""


# Setup
from sklearn import datasets
from pandas import DataFrame
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

# -------------------- Chapter 3

# Split data_setup into training and testing sets
from sklearn.cross_validation import train_test_split

# Scaling of features
from sklearn.preprocessing import StandardScaler

# Train perceptron and display output of algorithm on test set (Section 3)
from sklearn.linear_model import Perceptron

# Run performance metrics on algorithm
from sklearn.metrics import accuracy_score

# Logistic regression
from sklearn.linear_model import LogisticRegression

# Maximum margin classifier (suppert vector machine)
from sklearn.svm import SVC

# SGDClassifier for large datasets and partial fitting (online learning)
from sklearn.linear_model import SGDClassifier

# Solving nonlinear problems using kernel SVMs by specifying kernel in SVC

# Decision tree algorithm
from sklearn.tree import DecisionTreeClassifier

# Export trained model to graphviz
from sklearn.tree import export_graphviz

# Random forest
from sklearn.ensemble import RandomForestClassifier

# k-nearest neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier

StandardScaler(); Perceptron(); accuracy_score(); LogisticRegression(); SVC();
DecisionTreeClassifier(); export_graphviz(); RandomForestClassifier(); KNeighborsClassifier()

# --------------------- Chapter 4

# Imputation of missing values
from sklearn.preprocessing import Imputer

# Can use label encoder in scikit-learn
from sklearn.preprocessing import LabelEncoder

# One-hot encoder
from sklearn.preprocessing import OneHotEncoder

# Normalisation using scitkit-learn scaler (Section 4)
from sklearn.preprocessing import MinMaxScaler

# Feature selection using random forests using forest.feature_impotances_

Imputer(); LabelEncoder(); OneHotEncoder(); MinMaxScaler()

# ---------------------- Chapter 5

# Implement PCA
from sklearn.decomposition import PCA

# Implement LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Implement kernel PCA
from sklearn.decomposition import KernelPCA

PCA(); LDA(); KernelPCA()

# ---------------------- Chapter 6

# Pipeline several steps
from sklearn.pipeline import Pipeline

# Stratified k-fold cross-validation (stratified means that class proportions are preserved in cross validation splits)
from sklearn.cross_validation import StratifiedKFold

# More automated implementation of k-fold cross-validation:     scores = cross_val_score()
from sklearn.cross_validation import cross_val_score

# Use learning curve function from scikit-learn to evaluate the model:     train_sizes, train_scores, test_scores = learning_curve()
from sklearn.learning_curve import learning_curve

# Use validation curves to calibrate hyperparameters (e.g. l2 penalty parameter):       train_scores, test_scores = vaidation_curve
from sklearn.learning_curve import validation_curve

# Tuning hyperparamaters via grid search
from sklearn.grid_search import GridSearchCV # can also use RandomizedSearcgCV if we want to subsample among the parameter set

# Nested cross-validation using cross_val_score with estimator = GridSearchCV()

# Confusion matrix
from sklearn.metrics import confusion_matrix

# Display matrix as an image using ax.matshow, where fig, ax = plt.subplots(). Use e.g. cmap = plt.cm.Blues for blue colourmap

# Check different performance metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Specify different positive labels, can use scoring = makescorer() in GridSearchCV()
from sklearn.metrics import make_scorer

# Receiver operating characeristic (ROC) graph:    fpr, tpr, thresholds = roc_curve()
from sklearn.metrics import roc_curve

# Area under the curve (AUC)
from sklearn.metrics import auc

# Directly Compute AUC of ROC curve
from sklearn.metrics import roc_auc_score

# Different methods of averaging performance metrics in multiclass prediction by specifying average parameter in make_scorer
Pipeline(); StratifiedKFold(); cross_val_score(); learning_curve(); validation_curve(); GridSearchCV(); confusion_matrix();
precision_score(); recall_score(); f1_score(); make_scorer(); roc_curve(); auc(); roc_auc_score()

# ---------------------- Chapter 7

# Template for estimator
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
from itertools import product
from sklearn.ensemble import BaggingClassifier
from  sklearn.ensemble import AdaBoostClassifier