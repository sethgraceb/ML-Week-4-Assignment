#Q(ii)(d) kNN classifier with ROC final2
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
for train, test in kf.split(X2):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3,weights='uniform').fit(X2[train], y[train])
    X2test=np.linspace(0.0, 1.0).reshape(-1, 1)
    ypred = model.predict(X2test)
    model2 = KNeighborsClassifier(n_neighbors=7,weights='uniform').fit(X2[train], y[train])
    ypred2 = model2.predict(X2test)
    
    #Baseline
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent").fit(X2[train], y[train])
    ydummy = dummy.predict(X2[test])
    
    #ROC curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    preds = model.predict_proba(X2[test])
    print(model.classes_)
    fpr, tpr, _ = roc_curve(y[test],preds[:,1])
    plt.plot(fpr,tpr, label ='kNN classifier')
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1).fit(X2[train], y[train])
    fpr, tpr, _ = roc_curve(y[test],model.decision_function(X2[test]))
    plt.plot(fpr,tpr,color='orange', label = 'Baseline Classifier')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    
    plt.plot([0, 1], [0, 1], color='green',linestyle='--')
    plt.title("ROC Curve kNN Classifier")
    plt.legend()
    plt.show()
