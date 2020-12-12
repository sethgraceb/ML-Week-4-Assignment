#Q(i)(c) kNN classifier confusion matrix final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
df = pd.read_csv("week4.csv", comment = '#')

#feature values
X1 = np.array(df.iloc[:, 0])  #1st input feature
X1 = X1.reshape(-1, 1)

X2 = np.array(df.iloc[:, 1])  #2nd input feature
X2 = X2.reshape(-1, 1)
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])   #3rd column +/- target value
y = y.reshape(-1, 1).ravel()

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)      #cross validation
for train, test in kf.split(X1):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3,weights='uniform').fit(X1[train], y[train])
    X1test=np.linspace(0.0, 1.0).reshape(-1, 1)
    ypred = model.predict(X1test)
    model2 = KNeighborsClassifier(n_neighbors=7,weights='uniform').fit(X1[train], y[train])
    ypred2 = model2.predict(X1test)
    
   #confusion matrix
    preds = model.predict(X1[test])

    print("kNN Classifier:")
    from sklearn.metrics import confusion_matrix
    print("Confusion Matrix:\n",confusion_matrix(y[test], preds))
    from sklearn.metrics import classification_report
    print("Classification Report:\n", classification_report(y[test], preds))
    
    print("Baseline Classifier:")
    from sklearn.svm import LinearSVC
    model = LinearSVC(C=1.0).fit(X1[train], y[train])
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent").fit(X1[train], y[train])
    ydummy = dummy.predict(X1[test])
    print("Confusion Matrix:\n", confusion_matrix(y[test], ydummy))
    print("Classification Report: \n",classification_report(y[test], ydummy))
