import pickle
import os


def pickle_loader(input):
    item = pickle.load(open(input, 'rb'))
    return item.values


if __name__ == '__main__':

    training_file = ".\\traffic-signs-data\\train.p"
    validation_file = ".\\traffic-signs-data\\valid.p"
    testing_file = ".\\traffic-signs-data\\test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
