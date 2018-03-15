import itertools
import pickle

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree


def initialize_tree(settings):
    return tree.DecisionTreeClassifier(
        random_state=11, max_depth=settings[0], max_features=settings[1],
        criterion=settings[2], splitter=settings[3],
        min_samples_leaf=settings[4], min_samples_split=settings[5]
    )


def choose_setting(data):
    max_depth = [50, 100, 150, 200, 250, 300, 400, 500]
    max_features = [50, 100, 300, 1000, data[0][0].shape[1]]
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    min_samples_leaf = [1, 2, 3, 5, 7, 10]
    min_samples_split = [2, 3, 5, 7, 10]

    setting_options = itertools.product(
        max_depth, max_features, criterion, splitter, min_samples_leaf,
        min_samples_split
    )
    best_settings = []
    best_acc = -np.inf

    for settings in setting_options:
        print('Testing {:4}, {:4}, {:7}, {:6}, {:2}, {:2} '.format(*settings),
              end='')
        clf = initialize_tree(settings)
        clf.fit(data[0][0], data[0][1])
        accuracy = np.mean(clf.predict(data[1][0]) == data[1][1])
        print('\t\t\tAchieved Accuracy = {:%}'.format(accuracy))
        if accuracy > best_acc:
            best_settings = [settings]
            best_acc = accuracy
        elif accuracy == best_acc:
            best_settings.append(settings)

    return best_settings


def display_max_depth_effect(data, settings):
    train_accuracy = []
    valid_accuracy = []
    depths = sorted(list(set(np.exp2(np.arange(100) / 12).astype(int))))
    settings = list(settings)
    for d in depths:
        settings[0] = d
        clf = initialize_tree(settings)
        clf.fit(data[0][0], data[0][1])
        train_accuracy.append(np.mean(clf.predict(data[0][0]) == data[0][1]))
        valid_accuracy.append(np.mean(clf.predict(data[1][0]) == data[1][1]))

    plt.plot(depths, train_accuracy)
    plt.plot(depths, valid_accuracy)
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.legend(('Training', 'Validation'))
    plt.savefig('q7.png')
    plt.show()


def build_model(data, settings_loc='decision_tree_settings.pkl'):
    try:
        with open(settings_loc, 'rb') as f:
            settings = pickle.load(f)
    except IOError:
        settings = choose_setting(data)
        with open(settings_loc, 'wb') as f:
            pickle.dump(settings, f)

    print('Best settings are:', end='\n\t')
    print(*settings, sep='\n\t')
    print('Please enter the number coressponding to your choice:', end='\n\t')
    settings = settings[int(input())]

    clf = initialize_tree(settings)
    clf.fit(data[0][0], data[0][1])

    accuracy = np.mean(clf.predict(data[0][0]) == data[0][1])
    print('Training Accuracy: {:f}'.format(accuracy))
    accuracy = np.mean(clf.predict(data[1][0]) == data[1][1])
    print('Validation Accuracy: {:f}'.format(accuracy))
    accuracy = np.mean(clf.predict(data[2][0]) == data[2][1])
    print('Testing Accuracy: {:f}'.format(accuracy))

    display_max_depth_effect(data, settings)

    return clf


def display_graph(clf, keywords, max_depth=2):
    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=max_depth,
                                    feature_names=keywords,
                                    class_names=['real', 'fake'], rounded=True,
                                    filled=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")


def mutual_information_to_output(data, word):
    def entropy(a, condition=None):
        if condition is not None and sum(condition):
            a = a[condition]
        pa = np.mean(a)
        if pa == 0 or pa == 1:
            return 0
        return -pa * np.log2(pa) - (1 - pa) * np.log2(1 - pa)

    present = data[0].T[word].toarray().astype(bool)[0]
    not_present = np.logical_not(present)
    label = data[1]

    return entropy(label) - \
           np.mean(present) * entropy(label, present) - \
           np.mean(not_present) * entropy(label, not_present)