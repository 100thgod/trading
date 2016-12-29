
# Handing missing data
import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,,8.0
10.0, 11.0, 12.0,'''
df = pd.read_csv(StringIO(csv_data))
df.isnull().sum()
np_array = df.values
# Can drop rows with na values using dropna() method of pandas dataframes
df.dropna()
df.dropna(axis=1)
# Can only drop rows for which all columns are NaN
df.dropna(how = 'all')
# Can specifiy minimum number of non-NaNs for rows
df.dropna(thresh = 4)
# Only drops rows for which specific columns are NaN
df.dropna(subset = ['C'])

# Imputation of missing values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr.fit(df)
imputed_data = imr.transform(df.values)

# Categorical data
import pandas as pd
df = pd.DataFrame([
    ['greem', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']
# convert categorical features into integers
size_mapping = {
    'XL' : 3,
    'L' : 2,
    'M' : 1
}
df['size'] = df['size'].map(size_mapping)
# mapping for transform back into categorical labels
inv_size_mapping = {v : k for k,v in size_mapping.items()}

# Encoding class labels
import numpy as np
class_mapping = {label : idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
inv_class_mapping = {v : k for k,v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

# Can use label encoder in scikit-learn
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
# Does fitting and transforming
y = class_le.fit_transform(df['classlabel'].values)
# Does inverse transform, assumes that class_le is already fitted
class_le.inverse_transform(y)

# One-hot encoder
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

# Can use getDummies method for pandas dataframes
pd.get_dummies(df[['price','color','size']])

# Load wine dataset
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', \
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
print('Class labels', np.unique(df_wine['Class label']))
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.3, random_state = 0)

# Normalisation using scikit-learn scaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Standardisation using  scikit-learn scaler
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Feature selection using penalisation function
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l1', C = 0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
lr.intercept_
lr.coef_

# Plot regularisation path
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty = 'l1', C = 10**c, random_state = 0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:,column], label = df_wine.columns[column+1], color = color)
plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc = 'upper center', bbox_to_anchor = (1.38, 1.03), ncol = 1, fancybox = True)

# Feature selection using KNN classifier
from feature_selection import SBS
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = SBS(knn, k_features = 1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = 'o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# Print minimal set of features such that accuracy = 1
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
# Evaluate KNN on original dataset
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))
# Evaluate KNN with reduced feature space on original dataset
knn.fit(X_train_std[:,k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:,k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:,k5], y_test))

# Feature selection using random forests
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
forest.fit(X_train, y_train)
importances =  forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%.2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color = 'lightblue', align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()

# Can do feature selection with specified threshold using transform method of random forest object
X_selected = forest.transform(X_train, threshold = 0.15)