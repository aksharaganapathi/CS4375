import pandas as pd
import numpy as np

training_data = pd.read_csv('mush_train.data', header=None)
test_data = pd.read_csv('mush_test.data', header=None)

training_data.columns = ['category', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                         'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                         'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                         'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                         'spore-print-color', 'population', 'habitat']

test_data.columns = ['category', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                     'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                     'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                     'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                     'spore-print-color', 'population', 'habitat']


class TreeNode(object):
    def __init__(self, data, parent_attr_value):
        self.data = data
        self.parent_attr_value = parent_attr_value
        self.attr = None
        self.information_gain = None
        self.majority_vote = None
        self.decision = None
        self.parent_attr = None
        self.children = []


def entropy(data, target_attribute):
    total_samples = len(data)
    unique_labels, counts = np.unique(data[target_attribute], return_counts=True)
    proportions = counts / total_samples

    p_count = counts[unique_labels == 'p'][0] if 'p' in unique_labels else 0
    e_count = counts[unique_labels == 'e'][0] if 'e' in unique_labels else 0

    entropy_val = -np.sum(proportions * np.log2(proportions))

    return total_samples, p_count, e_count, entropy_val


def calculate_attribute_entropy(data, attribute, target_attr, total_entropy):
    attribute_values, counts = np.unique(data[attribute], return_counts=True)
    attribute_entropy = 0

    for value, count in zip(attribute_values, counts):
        subset_entropy = entropy(data[data[attribute] == value], target_attr)[3]
        attribute_entropy += (count / len(data)) * subset_entropy

    return attribute_entropy


def best_attribute(data, target_attr):
    total_samples, total_positive, total_negative, total_entropy = entropy(data, target_attr)

    max_information_gain = 0
    best_attr = None

    for attr in data.columns:
        if attr == target_attr:
            continue

        attr_entropy = calculate_attribute_entropy(data, attr, target_attr, total_entropy)

        information_gain = total_entropy - attr_entropy

        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attr = attr

    return best_attr, max_information_gain


def calculate_majority_vote(data, target_attr):
    num_positive_samples = np.sum(data[target_attr] == 'p')
    num_negative_samples = np.sum(data[target_attr] == 'e')
    majority_vote = 'p' if num_positive_samples > num_negative_samples else 'e'
    return majority_vote


def build_decision_tree(node, target_attr):
    data = node.data
    total_samples, num_features = data.shape
    majority_vote = calculate_majority_vote(data, target_attr)

    target_values = data[target_attr]
    unique_target_values, target_counts = np.unique(target_values, return_counts=True)

    if len(unique_target_values) == 1 or num_features == 1:
        node.majority_vote = majority_vote
        node.decision = majority_vote
        return

    best_attr, information_gain = best_attribute(data, target_attr)

    node.attr = best_attr
    node.information_gain = information_gain

    attr_values = np.unique(data[best_attr])

    for attr_value in attr_values:
        subset = data[data[best_attr] == attr_value].drop(columns=[best_attr])
        child_node = TreeNode(subset, attr_value)
        child_node.parent_attr = best_attr
        node.children.append(child_node)
        build_decision_tree(child_node, target_attr)



def decision_tree(node, target_attr):
    build_decision_tree(node, target_attr)


def accuracy(predicted, actual):
    predicted = np.array(predicted)
    actual = np.array(actual)

    if len(predicted) != len(actual):
        raise ValueError("Length of predicted and actual arrays must be the same.")

    correct_predictions = np.sum(predicted == actual)
    total_predictions = len(actual)

    if total_predictions == 0:
        return 0.0

    accuracy_value = correct_predictions / total_predictions
    return accuracy_value


def predict_classifier(node, data_point, level):
    if node.decision is not None:
        return node.decision

    attr = node.attr
    if attr not in data_point.index.values:
        if node.majority_vote != data_point['p']:
            return node.majority_vote

    child = None
    for child in node.children:
        if child.parent_attr_value == data_point[attr]:
            break

    return predict_classifier(child, data_point, level + 1)


def predict_and_evaluate_classifier(node, data, y_attr):
    predictions = []
    correct_predictions = 0

    for i in range(len(data)):
        prediction = predict_classifier(node, data.iloc[i], 0)
        predictions.append(prediction)

        if prediction == data[y_attr].iloc[i]:
            correct_predictions += 1

    accuracy_score = correct_predictions / len(data)
    print('Accuracy:', str(accuracy_score * 100) + "%")

    return predictions, accuracy_score


tree_node = TreeNode(training_data, None)
decision_tree(tree_node, 'category')
predict_and_evaluate_classifier(tree_node, test_data, 'category')
