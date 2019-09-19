import numpy as np

# fetch dataset:
from sklearn.datasets import fetch_openml

#mnist = fetch_openml('mnist_784', version=1)
mnist = fetch_openml('mnist_784', cache=True, version=1)
mnist.keys()

#iris = load_iris()
#X = iris.data[:, (2, 3)] # petal length, petal width
#y = (iris.target == 0).astype(np.int) # Iris Setosa?

X, y = mnist["data"], mnist["target"]


import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


y = y.astype(np.uint8)


# putting the image files into variables for test set and training (which has already been set up automatically):
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# TRAINING BINARY CLASSIFIER:
y_train_5 = (y_train == 5)
# 5 vs not-5: True for 5, False for all other digits.
y_test_5 = (y_test == 5)


# PICKING/TRAINING A CLASSIFIER (and training):
# Stochastic Gradient Descent (SGD) classifier

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

# -------------------------------------------
# performance eval (cross-validation and accuracy):

from sklearn.model_selection import cross_val_score

print('accuracy determined from cross validation: ' + str(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")))
print('precision determined from cross validation: ' + str(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="precision")))
print('recall determined from cross validation: ' + str(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="recall")))

# a BETTER way to eval than cross-validation: CONFUSION MATRIX:
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)


# -------------------------------------------
# Deciding on a Threshold value to determine how/where classes get divided up by examining the precision/recall Tradeoff:

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

y_scores = sgd_clf.decision_function([some_digit])

threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred



# get decision scores data to compute precision and recall for all possible thresholds-- 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# --by way of precision_recall_curve() function:

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# -------------------------------------------

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = (y_scores >= threshold_90_precision)

print('precision score calculated using 90 percent precision classifier: ' + str(precision_score(y_train_5, y_train_pred_90)))
print('recall score calculated using 90 percent precision classifier: ' + str(recall_score(y_train_5, y_train_pred_90)))

#...We have now finished construction of a 90% precision classifier.

# -------------------------------------------



