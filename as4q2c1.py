#Q(ii)(c) - logistic regression final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (10.0, 5.0)
df = pd.read_csv("week4.csv", comment = '#')
#feature values
X1 = np.array(df.iloc[:, 0])  #1st input feature
X1 = X1.reshape(-1, 1)
X2 = np.array(df.iloc[:, 1])  #2nd input feature
X2 = X2.reshape(-1, 1)
y = df.iloc[:, 2]   #3rd column +/- target value
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)   #cross validation

from sklearn.preprocessing import PolynomialFeatures
Xpoly = PolynomialFeatures(6).fit_transform(X2)
C_range = [0.5, 1, 5, 10, 50, 100]    #C values
for C in C_range:
    from sklearn.linear_model import Ridge
    model = Ridge(alpha = 1/(2 * C))
    temp = []; plotted = False
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])
        ypred = model.predict(Xpoly[test])
        
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.svm import LinearSVC
        model = LinearSVC(C=1.0).fit(Xpoly[train], y[train])
        
        print("\nLogistic Regression Classifier:")
        preds = model.predict(Xpoly[test])
        print("Confusion Matrix:\n",confusion_matrix(y[test], preds))
        print("Classification Report:\n", classification_report(y[test], preds))
        
        print("\nBaseline Classifier:")
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent").fit(X2[train], y[train])
        ydummy = dummy.predict(X2[test])
        print("Confusion Matrix:\n", confusion_matrix(y[test], ydummy))
        print("Classification Report: \n",classification_report(y[test], ydummy))
