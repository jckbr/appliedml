import csv
import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

def loadData(csvPath):
    with open(csvPath, 'r') as path:
        tempHeaders = csv.reader(path, delimiter = ';')
        allHeaders = next(tempHeaders)
        headersOut = allHeaders[:-1 or None]
        targetName = allHeaders[-1]

    fullData = numpy.genfromtxt(csvPath, delimiter = ';', skip_header = 1)
    csvData = fullData[:, :-1]
    dataTarget = fullData[:, -1]

    return csvData, dataTarget, headersOut, targetName, csvPath

redData, redTarget, redHeaders, redTargetName, redFileName = loadData('data/winequality-red.csv')

print("a) Data shape:\n", redData.shape, "\n")
print("b) Data feature description:\nvarious attributes about the makeup of wine\n")
print("c) Data target description:\n0 (lowest) to 10 (highest) quality of wine\n")
print("d) Data, first five rows:\n", redData[:5, :], "\n")
print("e) 3 Influential Features:\n")

citricAcidData = redData[:, 2]
pHData = redData[:, 8]
alcoholData = redData[:, 10]
colorValues = numpy.arange(0, 11)

fig, axes = plt.subplots(1, 3, sharey = 'all', tight_layout = True)
axes[0].hist(citricAcidData, bins = 9)
axes[0].title.set_text("Citric Acid")
axes[1].hist(pHData, bins = 9)
axes[1].title.set_text("pH")
axes[2].hist(alcoholData, bins = 9)
axes[2].title.set_text("Alcohol")
fig.suptitle("Three Influential Features")
plt.show()

print("f) Features against Target:\n")
fig, axes = plt.subplots(1, 3, tight_layout = True)
axes[0].scatter(x = citricAcidData, y = pHData, c = redTarget, cmap = 'rainbow')
axes[0].title.set_name("Citric Acid against pH")
axes[0].set_xlabel("Citric Acid")
axes[0].set_ylabel("pH")
axes[1].scatter(x = pHData, y = alcoholData, c = redTarget, cmap = 'rainbow')
axes[1].title.set_name("pH against Alcohol")
axes[1].set_xlabel("pH")
axes[1].set_ylabel("Alcohol")
axes[2].scatter(x = alcoholData, y = citricAcidData, c = redTarget, cmap = 'rainbow')
axes[2].title.set_name("Alcohol against Citric Acid")
axes[2].set_xlabel("Alcohol")
axes[2].set_ylabel("Citric Acid")
fig.suptitle("Features against Target data")
plt.show()

kNNneighbors = 30
print(f"4) KNN with n_neighbors from 1 to {kNNneighbors}:\n")
trainScore = []
testScore = []
X_train, X_test, y_train, y_test = train_test_split(redData, redTarget)
for i in range(1, kNNneighbors):
    knnModel = KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train)
    trainScore.append(knnModel.score(X_train, y_train))
    testScore.append(knnModel.score(X_test, y_test))

plt.plot(range(1, kNNneighbors), trainScore, label ="Train accuracy %")
plt.plot(range(1, kNNneighbors), testScore, label ="Test accuracy %", linestyle ='dashed')
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.title("Train / Test Accuracy of kNN with varying neighbor count")
plt.legend()
plt.show()

print("5) KNN with 5-fold-shuffle cross validation:\n")
kFvals = []
kFold = KFold(n_splits = 5, shuffle = True, random_state = 9)
for i in range(1, kNNneighbors):
    knnModelNoFit = KNeighborsClassifier(n_neighbors = i)
    output = cross_validate(knnModelNoFit, redData, redTarget, cv = kFold, return_train_score = True)
    kFvals.append([i, numpy.mean(numpy.array(output["train_score"])), numpy.mean(numpy.array(output["test_score"]))])

summarizedScore = pandas.DataFrame(kFvals, columns = ['kNN', 'train_score', 'test_score'])
plt.plot(summarizedScore['kNN'], summarizedScore['train_score'], label = "Mean train score")
plt.plot(summarizedScore['kNN'], summarizedScore['test_score'], label = "Mean test score", linestyle = 'dashed')
plt.xlabel("n_neighbors")
plt.ylabel("Score")
plt.title("5-Fold-Shuffle Cross Validation of kNN")
plt.legend()
plt.show()

print("6) Prediction accuracy of two rows of misc data:")
allData = numpy.genfromtxt('data/winequality-red-test.csv', delimiter = ';', skip_header = 1)
testData = allData[:, :-1]
testTargetData = allData[:, -1]
testPredictions = knnModel.predict(testData)

testAccuracy = accuracy_score(testPredictions, testTargetData)
print(testAccuracy)