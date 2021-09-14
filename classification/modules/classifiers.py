from sklearn import tree
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# import the regressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

##Metrics
from sklearn.metrics import mean_squared_error

from sklearn import svm
from sklearn import naive_bayes


def split_dataset_train(X, y, n_train_percent):
    # split train/test data set
    n_data = len(X)
    n_train = int(n_train_percent * n_data / 100)

    X_train = X[1:n_train]
    y_train = y[1:n_train]

    return X_train, y_train

def split_dataset_test(X, y, n_train_percent):
    # split train/test data set
    n_data = len(X)
    n_train = int(n_train_percent * n_data / 100)

    X_test = X[n_train + 1:]
    y_test = y[n_train + 1:]

    return X_test, y_test

def create_svm():
    model = svm.SVC()
    return model

def create_svm_multiclass():
    # one vs the rest
    model = svm.LinearSVC()
    return model

def create_naive_bayes():
    # model = naive_bayes.BernoulliNB()
    model = naive_bayes.MultinomialNB()
    return model

def create_decision_tree():
    model = DecisionTreeClassifier()
    return model

def create_multi_output_classifier(use_randomforest):
    if use_randomforest:
        # model = MultiOutputClassifier(RandomForestClassifier())
        model = RandomForestClassifier()
    else:
        # model = MultiOutputClassifier(DecisionTreeClassifier())
        model = DecisionTreeClassifier()
    
    return model


def train_decision_tree(model, X_train, y_train):
    model.fit(X_train, y_train)

    # check training accuracy
    # predict the training data set using the model
    y_predict = model.predict(X_train)

    y_diff = y_train - y_predict

    print("target (train): ")
    print(y_train)
    print("prediction (train): ")
    print(y_predict)
    print("diff (train): ")
    print(y_diff)

    y_diff = np.reshape(y_diff, (1,-1))
    # print(np.shape(y_diff))
    y_diff = y_diff[0]
    # print(y_diff)
    
    accuracy = len([yd for yd in y_diff if yd == 0]) / len(y_diff) * 100

    # accuracy = mean_squared_error(y_train, y_predict)

    print("accuracy: " + str(round(accuracy, 2)) + " %")

    return model, accuracy


def predict_decision_tree(model, X_test, y_test, verbose):

    # predict the test data set using the model
    y_predict = model.predict(X_test)

    y_predict_a = np.array(y_predict)
    y_test = np.array(y_test)
    y_diff = y_test - y_predict_a

    y_diff = np.reshape(y_diff, (1,-1))
    # print(np.shape(y_diff))
    y_diff = y_diff[0]
    # print(y_diff)

    ndiff = len([yd for yd in y_diff if yd == 0])
    accuracy = ndiff / len(y_diff) * 100

    # accuracy = mean_squared_error(y_test, y_predict)

    if verbose:
        print("target (test): ")
        print(y_test)
        print("prediction (test): ")
        print(y_predict_a)
        print("diff (test): ")
        print(y_diff)
        print("accuracy: " + str(round(accuracy, 2)) + " %")

    return model, accuracy, ndiff, len(y_diff), y_predict
