import pickle


def save_sklearn_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_sklearn_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
