# import modules
from loader import load_dataset
from disp_utils import print_decision_tree, plot_decision_tree
from classifiers import create_decision_tree, create_random_forest

input_file = "./data/winequality-white.csv"
X, y, features, classes = load_dataset(input_file, False)

X = X[:100]
y = y[:100]

n_train_percent = 80

# train decision tree classifier
model1, accuracy1 = create_decision_tree(X, y, n_train_percent)
model2, accuracy2 = create_random_forest(X, y, n_train_percent)

print("decision tree prediction accuracy: ", accuracy1)
print("random forest prediction accuracy: ", accuracy2)

plot_decision_tree(model1, features, classes, "dtree31.png")

print("done")


