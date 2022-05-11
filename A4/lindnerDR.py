import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, plot_roc_curve
pandas.set_option('display.max.colwidth', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

# 1) Load dataset
housePricesRaw = pandas.read_csv(open('houseSalePrices.csv' , 'r', encoding = 'UTF-8'), index_col = 'Id')

# 2) Meet the data and select three influential features
print("House prices data shape:", housePricesRaw.shape)
print("Data features:", housePricesRaw.columns.values[:-3])
print("Data target:", housePricesRaw.columns.values[-1:])

housePrices = housePricesRaw[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal']].copy()
housePricesTarget = housePricesRaw[['SalePrice']].copy()

LotArea = housePrices['LotArea']
YearBuilt = housePrices['YearBuilt']
GarageCars = housePrices['GarageCars']

fig, axes = plt.subplots(1, 3, sharey = 'all', tight_layout = True)
axes[0].hist(LotArea, bins = 5)
axes[0].title.set_text("Lot Area")
axes[1].hist(YearBuilt, bins = 5)
axes[1].title.set_text("Year Built")
axes[2].hist(GarageCars, bins = 4)
axes[2].title.set_text("Garage Cars")
fig.suptitle("Three Influential Features")
plt.show()

# 3) Select and fill missing data
fillNaNMean = list(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
for col in fillNaNMean:
    housePrices[col].fillna(housePrices[col].mean(), inplace = True)

# 4) Remove low correlation columns (30%)
corr = housePricesRaw.corr()
relevantFeatures = corr.nlargest(int(corr.shape[0] * (1.0 - 0.3)), 'SalePrice').index
housePricesRelevant = housePricesRaw[relevantFeatures].copy()
relevantCorr = housePricesRelevant.corr()
seaborn.heatmap(relevantCorr, annot = True, cmap = plt.cm.jet)
print("Features after removing 30%:", relevantCorr.shape[0])
plt.show()

# 5) Convert columns using category conversion
convertToNum = housePricesRelevant.columns[housePricesRelevant.isnull().any()]
for col in convertToNum:
    housePricesRelevant[col] = housePricesRelevant[col].astype('category').cat.codes

# 6) Use linear regression as base, keep only top 50% of features
X_train, X_test, y_train, y_test = train_test_split(housePricesRelevant, housePricesTarget, test_size = 0.2)
linReg = LinearRegression().fit(X_train, y_train)
linRegTrainScore = linReg.score(X_train, y_train)
linRegTestScore = linReg.score(X_test, y_test)
print("Linear Regression train score: {:.4f}".format(linRegTrainScore))
print("Linear Regression test score: {:.4f}".format(linRegTestScore))

selPer = SelectPercentile(f_classif, percentile = 50)
selPerFit = selPer.fit_transform(housePricesRelevant, housePricesTarget.values.ravel())
cols = selPer.get_support(indices = True)
selPerDF = housePricesRelevant.iloc[:, cols].copy()
selPerDF = selPerDF[selPerDF.columns[::-1]]

selPerDFCorr = selPerDF.corr()
seaborn.heatmap(selPerDFCorr, annot = True, cmap = plt.cm.turbo)
print("Features after keeping 50% from SelectPercentile:", selPerDFCorr.shape[0])
plt.show()

selPerData = selPerDF.iloc[:, :-1]
selPerTarget = selPerDF.iloc[:, -1]

# 7) Use PCA on remaining features, PCA means 10% of remaining features
pca = PCA(n_components = int(selPerData.shape[1] * 0.1))
pcaFit = pca.fit_transform(selPerData, selPerTarget)
pcaDF = pandas.DataFrame(pca.components_, columns = selPerData.columns)
print("Shape of new dataframe after PCA:", pcaDF.shape)

pcaCols = pcaDF.columns.values
pcaData = housePrices[pcaCols].copy()

# 8) Use non-linear model like SVM, Tree based, NN, etc with parameter tuning
X_train, X_test, y_train, y_test = train_test_split(pcaData, selPerTarget, test_size = 0.2)

svcSVM = SVC(kernel = 'rbf')
svcSVM.fit(X_train, y_train)
svcSVMTrain = svcSVM.score(X_train, y_train)
svcSVMTest = svcSVM.score(X_test, y_test)
svcSVMPred = svcSVM.predict(X_test)
svcSVMPrec = numpy.mean(precision_score(y_test, svcSVMPred, average = None))
svcSVMRecall = numpy.mean(recall_score(y_test, svcSVMPred, average = None))

regTree = DecisionTreeRegressor()
regTree.fit(X_train, y_train)
regTreeTrain = regTree.score(X_train, y_train)
regTreeTest = regTree.score(X_test, y_test)
regTreePred = regTree.predict(X_test)

randForest = RandomForestRegressor(n_estimators = 50)
randForest.fit(X_train, y_train)
randForestTrain = randForest.score(X_train, y_train)
randForestTest = randForest.score(X_test, y_test)
randForestPred = randForest.predict(X_test)

nonLinDF = pandas.DataFrame({'SVC':[svcSVMTrain, svcSVMTest, svcSVMPrec, svcSVMRecall], 'RegTree':[regTreeTrain, regTreeTest, numpy.NaN, numpy.NaN], 'RegForest':[randForestTrain, randForestTest, numpy.NaN, numpy.NaN]}, index = ['Train', 'Test', 'Precision', 'Recall'])
print(nonLinDF)