
import pandas as pd
import numpy as np
df = pd.read_excel('/Users/armtiger/Documents/Python/iris_dataset.xls')

# Encode labels
from sklearn.preprocessing import LabelEncoder
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Pipeline data standardisation, PCA and logistic regression steps
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components = 2)), ('clf', LogisticRegression(random_state = 1))])
pipe_lr.fit(X_train, y_train)
pipe_lr.score(X_test, y_test)

# Stratified cross_validation
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y = y_train, n_folds = 10, random_state = 1)
scores = []
for k, (train, test) in enumerate(kfold):
    print(k, train, test)
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist. %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
print('CV accuracy: %.3f +/i %.3f' % (np.mean(scores), np.std(scores)))

# Another implementation of k-fold cross-validation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator = pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1)
print('CV accuracy scores: %s' % scores)
print('CV accusracy: %.3f =/- %.3f' % (np.mean(scores), np.std(scores)))

# Use learning curve function from scikit-learn to evaluate the model
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(penalty = 'l2', random_state = 0))])
train_sizes, train_scores, test_scores = \
    learning_curve(estimator = pipe_lr, X = X_train, y = y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv = 10, n_jobs = 1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color = 'green', marker = 's', markersize = 5, label = 'testing accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color = 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.5, 1.0])
plt.show()

# address under-/overfitting with validation curves
from sklearn.learning_curve import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = \
    validation_curve(estimator = pipe_lr, X = X_train, y = y_train, param_name = 'clf__C', param_range = param_range, cv = 10)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)
plt.plot(param_range, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, color = 'blue', alpha = 0.15)
plt.plot(param_range, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color = 'green', alpha = 0.15)

# Tuning hyperparamaters via grid search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state = 1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C' : param_range, 'clf__kernel' : ['linear']}, {'clf__C' : param_range, 'clf__gamma' : param_range, 'clf__kernel' : ['rbf']}]
gs = GridSearchCV(estimator = pipe_svc, param_grid =  param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy : %.3f' % clf.score(X_test, y_test))

# Nested cross-validation
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = 'accuracy', cv = 2, n_jobs = -1)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 5)
# Use with decision tree classifier to tune depth parameter
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 0), \
        param_grid = [{'max_depth': [1,2,4,5,6,7,None]}], scoring = 'accuracy', cv = 5)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 2)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Confusion matrix
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confmat)
# display results using matplotlib's matshow function
fig, ax = plt.subplots(figsize = (2.5, 2.5))
ax.matshow(confmat, cmap = plt.cm.Blues, alpha = 0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x = j, y = i, s = confmat[i,j], va = 'center', ha = 'center')
plt.xlabel('predicted label')
plt.ylabel('true label')

# Check different performance metrics
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true = y_test, y_pred = y_pred))
print('Recall: %.3f' % recall_score(y_true = y_test, y_pred = y_pred))
print('F1: %.3f' % f1_score(y_true = y_test, y_pred = y_pred))

# Specify different positive labels
from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, pos_label = 0)
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = scorer, cv = 10)

# Plot receiver-operator the area under the curve of the receiver operating characteristic (ROC) graph
from sklearn.metrics import roc_curve, auc
from scipy import interp
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components = 1)), ('clf', LogisticRegression(penalty = 'l2', random_state = 0, C = 100.0))])
cv = StratifiedKFold(y_train, n_folds = 3)
fig = plt.figure(figsize = (7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label = 1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw = 1, label = 'ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
plt.plot([0,1], [0,1], linestyle = '--', color = (0.6, 0.6, 0.6), label = 'random guessing')
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label = 'mean ROC (area = %0.2f)' % mean_auc, lw = 2)
plt.plot([0, 0, 1], [0, 1, 1], lw = 2, linestyle = ':', color = 'black', label = 'perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc = 'lower right')

# Can directly use roc_auc_score function from scikit-learn
pipe_lr = pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
print('ROC_AUC: %.3f' % roc_auc_score(y_true = y_test, y_score = y_pred))

# Different methods of averaging performance metrics in multiclass prediction
pre_scorer = make_scorer(score_func = precision_score, pos_label = 1, greater_is_better = True, average = 'micro')
