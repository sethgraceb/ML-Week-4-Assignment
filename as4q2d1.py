#Q(ii)(d) - logistic regression ROC final
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
        
        #ROC curve
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1).fit(Xpoly[train], y[train])
        from sklearn.metrics import roc_curve
        preds = model.predict_proba(Xpoly[test])
        print(model.classes_)
        fpr, tpr, _ = roc_curve(y[test],preds[:,1])
        plt.plot(fpr,tpr, color = 'blue', label ='Logistic Regression Classifier')
    
        #Baseline
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1).fit(X2[train], y[train])
        fpr, tpr, _ = roc_curve(y[test],model.decision_function(X2[test]))
        plt.plot(fpr,tpr,color='orange', label = 'Baseline Classifier')
        
        plt.plot([0, 1], [0, 1], color='green',linestyle='--')
        plt.title("ROC Curve Logistic Regression Classifier")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()