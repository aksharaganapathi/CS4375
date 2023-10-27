import pandas as pd
import numpy as np

column_names = ['OVERALL_DIAGNOSIS'] + [f'F{i}' for i in range(1, 23)]
training_data = pd.read_csv('heart_train.data', names=column_names, header=None)
test_data = pd.read_csv('heart_test.data', names=column_names, header=None)

for data in [training_data, test_data]:
    data['OVERALL_DIAGNOSIS'].replace(0, -1, inplace=True)


class Node(object):

    def __init__(self, data, parent_attribute_value):
        self.data = data
        self.attribute = None
        self.decision = None
        self.parent_attribute = None
        self.parent_attribute_value = parent_attribute_value
        self.children = []


def find_optimal_attribute(node):
    dataset = node.data
    min_error = float('inf')
    optimal_attr = None
    is_zero_positive = None

    for attribute in dataset.columns:
        if attribute in ['OVERALL_DIAGNOSIS', 'W']:
            continue

        grouped = dataset.groupby([attribute, 'OVERALL_DIAGNOSIS'])['W'].sum().unstack(fill_value=0)
        error_when_zero_positive = ((grouped.loc[0, -1] if 0 in grouped.index else 0) +
                                    (grouped.loc[1, 1] if 1 in grouped.index else 0))

        error_when_one_positive = ((grouped.loc[0, 1] if 0 in grouped.index else 0) +
                                   (grouped.loc[1, -1] if 1 in grouped.index else 0))

        error = min(error_when_zero_positive, error_when_one_positive)

        if error < min_error:
            min_error = error
            optimal_attr = attribute
            is_zero_positive = error_when_zero_positive < error_when_one_positive

    total_positive_error = dataset.loc[dataset['OVERALL_DIAGNOSIS'] == -1, 'W'].sum()
    total_negative_error = dataset.loc[dataset['OVERALL_DIAGNOSIS'] == 1, 'W'].sum()

    all_positive = all_negative = False

    if total_positive_error < min_error:
        min_error = total_positive_error
        all_positive = True
        all_negative = False

    if total_negative_error < min_error:
        min_error = total_negative_error
        all_positive = False
        all_negative = True

    return optimal_attr, min_error, is_zero_positive, all_positive, all_negative


def construct_decision_tree(node):
    dataset = node.data
    optimal_attr, error, is_zero_positive, all_positive, all_negative = find_optimal_attribute(node)

    if all_positive:
        node.decision = 1
        node.attribute = '+'
        return error

    if all_negative:
        node.decision = -1
        node.attribute = '-'
        return error

    node.attribute = optimal_attr
    unique_values = dataset[optimal_attr].unique()

    for value in unique_values:
        subset = dataset[dataset[optimal_attr] == value]
        child_node = Node(subset, value)
        child_node.parent_attribute = optimal_attr
        node.children.append(child_node)

        if is_zero_positive:
            child_node.decision = 1 if value == 0 else -1
        else:
            child_node.decision = 1 if value == 1 else -1

    return error


class Tree(object):

    def __init__(self, data):
        self.data = data
        self.root = None
        self.error = None

    def classify(self):
        node = Node(self.data, None)
        self.root = node
        return construct_decision_tree(node)

    def predict_dataset(self, data):
        return [self.predict(self.root, data.iloc[i]) for i in range(len(data))]

    def predict(self, node, data_point):
        if node.decision is not None:
            return node.decision

        attribute = node.attribute

        child = next((child for child in node.children if child.parent_attribute_value == data_point[attribute]), None)

        if child:
            return self.predict(child, data_point)


def update_sample_weights(data, tree_classifier, alpha_value, classification_error):
    actual_diagnosis = data['OVERALL_DIAGNOSIS']
    predicted_diagnosis = tree_classifier.predict_dataset(data)

    exp_vals = np.exp(-actual_diagnosis * predicted_diagnosis * alpha_value)
    normalization_constant = 2 * np.sqrt(classification_error * (1 - classification_error))

    data['W'] *= exp_vals / normalization_constant

    return data


def adaboost(ada_data, num_iterations):
    num_samples, num_features = ada_data.shape
    data = ada_data.copy()
    data['W'] = np.full(num_samples, 1.0 / num_samples).reshape(num_samples, 1)

    classifiers = []
    alphas = []

    for i in range(num_iterations):
        tree_classifier = Tree(data)
        classification_error = tree_classifier.classify()

        alpha_value = 0.5 * np.log((1 - classification_error) / classification_error)

        data = update_sample_weights(data, tree_classifier, alpha_value, classification_error)

        alphas.append(alpha_value)
        classifiers.append(tree_classifier)

    return alphas, classifiers


def predict_diagnosis(data, alpha_values, classifier_list):
    num_samples, num_features = data.shape
    actual_diagnosis = data['OVERALL_DIAGNOSIS'].values.reshape(num_samples, 1)
    predicted_diagnosis = np.zeros(num_samples).reshape(num_samples, 1)

    num_classifiers = len(classifier_list)

    for i in range(num_classifiers):
        alpha = alpha_values[i]
        predicted_diagnosis += alpha * (np.array(classifier_list[i].predict_dataset(data)).reshape(num_samples, 1))

    predicted_diagnosis = np.sign(predicted_diagnosis)

    misclassification_count = np.sum(np.abs(actual_diagnosis - predicted_diagnosis)) / 2

    return (1.0 - misclassification_count / num_samples) * 100


alphas, classifiers = adaboost(training_data, 10)
print('Training Accuracy: ', predict_diagnosis(training_data, alphas, classifiers))
print('Test Accuracy : ', predict_diagnosis(test_data, alphas, classifiers))
