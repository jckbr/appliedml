import csv
import random
import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_validate

def loadData(csvpath):
    with open(csvpath, 'r') as path:
        data = csv.reader(path, delimiter = ",")
        allHeaders = next(data)
        targetHeader = allHeaders[-1]
        fullData = numpy.genfromtxt(csvpath, delimiter = ",", skip_header = 1)

    data = fullData[:, :-1]
    target = fullData[:, -1]

    return data, target, allHeaders, targetHeader

# Regression task
# a)
superconductData, superconductTarget, superconductHeaders, superconductTargetHeader = loadData('superconduct/train.csv')

# b)
numElements = superconductData[:, 0]
meanDensity = superconductData[:, 31]
meanAtomicRadius = superconductData[:, 22]

fig, axes = plt.subplots(1, 3, tight_layout = True)
axes[0].scatter(x = numElements, y = meanDensity, c = superconductTarget, cmap = 'rainbow')
axes[0].title.set_name("Element # against Mean Density")
axes[0].set_xlabel("Number of Elements")
axes[0].set_ylabel("Mean Density")
axes[1].scatter(x = meanDensity, y = meanAtomicRadius, c = superconductTarget, cmap = 'rainbow')
axes[1].title.set_name("Mean Density against Mean Atomic Radius")
axes[1].set_xlabel("Mean Density")
axes[1].set_ylabel("Mean Atomic Radius")
axes[2].scatter(x = meanAtomicRadius, y = numElements, c = superconductTarget, cmap = 'rainbow')
axes[2].title.set_name("Mean Atomic Radius against Element #")
axes[2].set_xlabel("Mean Atomic Radius")
axes[2].set_ylabel("Number of Elements")
fig.suptitle("Features against Target data")
plt.show()

# c)
X_train, X_test, y_train, y_test = train_test_split(superconductData, superconductTarget)

linReg = LinearRegression().fit(X_train, y_train)
linRegSlope = linReg.coef_
linRegInt = linReg.intercept_
print("Regression: Linear Regression modeled\n")

lasso = Lasso(alpha = 0.01, max_iter = 1000).fit(X_train, y_train)
print("Regression: Lasso (alpha=0.01) modeled\n")
lasso1 = Lasso(alpha = 0.1, max_iter = 1000).fit(X_train, y_train)
print("Regression: Lasso (alpha=0.1) modeled\n")
lasso2 = Lasso(alpha = 0.5, max_iter = 1000).fit(X_train, y_train)
print("Regression: Lasso (alpha=0.5) modeled\n")

ridge = Ridge(alpha = 10).fit(X_train, y_train)
print("Regression: Ridge (alpha=10) modeled\n")
ridge1 = Ridge(alpha = 1).fit(X_train, y_train)
print("Regression: Ridge (alpha=1) modeled\n")
ridge2 = Ridge(alpha = 0.1).fit(X_train, y_train)
print("Regression: Ridge (alpha=0.1) modeled\n")

kFold = KFold(n_splits = 5, shuffle = True, random_state = 9)
lassoKfold = Lasso(alpha = 0.01, max_iter = 2000).fit(X_train, y_train)
output = cross_validate(lassoKfold, superconductData, superconductTarget, cv = kFold, return_train_score = True)
print("Regression: Lasso (kFold 5, alpha=0.01) modeled\n")
# d/e)

randomRows = random.sample(range(X_test.shape[0]), 10)
xRandRows = X_test[numpy.array(randomRows), :]
yRandRows = y_test[randomRows]

r2 = []
rmse = []
predValues = []
values = numpy.empty((7, 10))

values[0] = lasso.predict(xRandRows)
values[1] = lasso1.predict(xRandRows)
values[2] = lasso2.predict(xRandRows)
values[3] = ridge.predict(xRandRows)
values[4] = ridge1.predict(xRandRows)
values[5] = ridge2.predict(xRandRows)
values[6] = linReg.predict(xRandRows)

for i in range(0, 7):
    r2.append(r2_score(yRandRows, values[i]))
    rmse.append(numpy.sqrt(mean_squared_error(yRandRows, values[i])))

r2 = numpy.array(r2)
r2 = numpy.vstack(r2)
rmse = numpy.array(rmse)
rmse = numpy.vstack(rmse)

array1 = numpy.append(r2, rmse, axis = 1)
finalArray = numpy.append(array1, values, axis = 1)
dataFrame = pandas.DataFrame(finalArray, index = ['Lasso-1', 'Lasso-2', 'Lasso-3', 'Ridge-1', 'Ridge-2', 'Ridge-3', 'Linear'], columns = ['R^2', 'RMSE', 'pred0', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'pred6', 'pred7', 'pred8', 'pred9'])
print("Regression DataFrame:\n", dataFrame, "\n")

# Classification task
# a)
pulsarData, pulsarTarget, pulsarHeader, pulsarTargetHeader = loadData('HTRUpulsarCandidates/HTRU_2.csv')

# b)
profileMean = pulsarData[:, 0]
DMSNRMean = pulsarData[:, 4]
DMSNRKurtosis = pulsarData[:, 6]

fig, axes = plt.subplots(1, 3, tight_layout = True)
axes[0].scatter(x = profileMean, y = DMSNRMean, c = pulsarTarget, cmap = 'rainbow')
axes[0].title.set_name("Profile against DMSNR")
axes[0].set_xlabel("Integrated Profile Mean")
axes[0].set_ylabel("DMSNR Curve Mean")
axes[1].scatter(x = DMSNRMean, y = DMSNRKurtosis, c = pulsarTarget, cmap = 'rainbow')
axes[1].title.set_name("DMSNR Mean against DMSNR Kurtosis")
axes[1].set_xlabel("DMSNR Curve Mean")
axes[1].set_ylabel("DMSNR Curve Excess Kurtosis")
axes[2].scatter(x =DMSNRKurtosis, y = profileMean, c = pulsarTarget, cmap = 'rainbow')
axes[2].title.set_name("DMSNR Kurtosis against Profile Mean")
axes[2].set_xlabel("DMSNR Curve Excess Kurtosis")
axes[2].set_ylabel("Integrated Profile Mean")
fig.suptitle("Features against Target data")
plt.show()

# c/d)
X_train, X_test, y_train, y_test = train_test_split(pulsarData, pulsarTarget)

logReg = LogisticRegression().fit(X_train, y_train)

# e)
predictions = logReg.predict(X_test)
accuracy = logReg.score(X_test, y_test)
print("Classification: LogReg score =", accuracy)