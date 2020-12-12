#Q(i)(b) - final with polynomial features
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
y = df.iloc[:, 2]   #3rd column +/- target value

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)      #cross validation

q_range = [1,2,3,4,5,6]
for q in q_range:
    from sklearn.preprocessing import PolynomialFeatures
    Xpoly = PolynomialFeatures(q).fit_transform(X1)
    for train, test in kf.split(Xpoly):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3,weights='uniform').fit(X1[train], y[train])
        X1test=np.linspace(0.0, 1.0).reshape(-1, 1)
        ypred = model.predict(X1test)
        model2 = KNeighborsClassifier(n_neighbors=7,weights='uniform').fit(X1[train], y[train])
        ypred2 = model2.predict(X1test)

        import matplotlib.pyplot as plt
        plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
        plt.scatter(X1, y, color='red', marker='+')
        plt.plot(X1test, ypred, color='green')
        plt.title("n_neighbors = 3")
        plt.xlabel("input x"); plt.ylabel("output y"); plt.legend(["predict","train"])
        plt.show()

        plt.scatter(X1, y, color='red', marker='+')
        plt.plot(X1test, ypred2, color='blue')
        plt.title("n_neighbors = 7")
        plt.xlabel("input x"); plt.ylabel("output y"); plt.legend(["predict","train"])
        plt.show()