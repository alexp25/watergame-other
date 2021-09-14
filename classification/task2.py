# import modules
from loader import load_dataset
from disp_utils import print_decision_tree, plot_decision_tree
from classifiers import create_decision_tree, create_random_forest

import matplotlib.pyplot as plt

input_file = "./data/winequality-white.csv"
X, y, features, classes = load_dataset(input_file, False)

# X = X[:100]
# y = y[:100]

acc_vect1 = []
acc_vect2 = []

n_train_percent_vect = range(10, 90, 5)

for n_train_percent in n_train_percent_vect:
    # train decision tree classifier
    model1, accuracy1 = create_decision_tree(X, y, n_train_percent)
    model2, accuracy2 = create_random_forest(X, y, n_train_percent)
    acc_vect1.append(accuracy1)
    acc_vect2.append(accuracy2)

plt.plot(n_train_percent_vect, acc_vect1, n_train_percent_vect, acc_vect2)
plt.xlabel("training data [%]")
plt.ylabel("prediction accuracy [%]")
plt.title("Training/prediction results")
plt.legend(["dtree", "randomforest"])
plt.show()


