#Q(i)(a) - final
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
mean_error=[]; std_error=[]
q_range = [1,2,3,4,5,6]
for q in q_range:
    from sklearn.preprocessing import PolynomialFeatures
    Xpoly = PolynomialFeatures(q).fit_transform(X1)
    C_range = [0.5, 1, 5, 10, 50, 100]    #C values
    for C in C_range:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha = 1/(2 * C))
        temp = []; plotted = False
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y[test], ypred)
            temp.append(mean_squared_error(y[test], ypred))
        
            if((q == 1) or (q == 2) or (q == 6)) and not plotted:
                plt.scatter(X1, y, color ='blue', label = 'training data')
                ypred = model.predict(Xpoly)
                plt.scatter(X1, ypred, color = 'orange', label = 'predictions')
                plt.xlabel("input X1"); plt.ylabel("output y")
                plt.title("2D plot")
                plt.legend()

                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection = '3d')
                ax.scatter(X1, X2, y, color = 'blue', label = 'training data')
                ax.scatter(X1, X2, ypred, color = 'orange', label = 'predictions')
                ax.set_xlabel("Feature 1", fontsize = 30, color = 'blue')
                ax.set_ylabel("Feature 2", fontsize = 30, color = 'blue')
                ax.set_title("3D plot")
                plt.legend()
                plt.show()
                plotted = True              
    mean_error.append(np.array(temp).std())  #mean
    std_error.append(np.array(temp).std())   #variance
plt.errorbar(C_range, mean_error, yerr = std_error)
plt.title('Fold = 5')
plt.xlabel('C'); plt.ylabel('Mean Square Error')
plt.legend(['Error Bar'])
plt.show()