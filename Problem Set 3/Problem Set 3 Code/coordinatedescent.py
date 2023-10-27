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


class Tree(object):

    def __init__(self, data, attribute, are_0_positive, are_all_positive):
        self.data = data
        self.root = None
        self.attribute = attribute
        self.are_0_positive = are_0_positive
        self.are_all_positive = are_all_positive
        self.training_error = None

    def classify(self):
        node = Node(self.data, None)
        self.root = node
        self.build_tree(node)

    def build_tree(self, node):
        data = node.data
        attribute = self.attribute
        is_0_positive = self.are_0_positive
        is_all_positive = self.are_all_positive

        if is_all_positive is not None:
            node.decision = 1 if is_all_positive else -1
            node.attribute = 'POSITIVE' if is_all_positive else 'NEGATIVE'
            return

        unique_attr_values = data[attribute].unique()
        node.attribute = attribute

        for attr_value in unique_attr_values:
            split_data = data[data[attribute] == attr_value]
            child = Node(split_data, attr_value)
            child.parent_attribute = attribute
            node.children.append(child)

            if attr_value == 0:
                child.decision = 1 if is_0_positive else -1
            elif attr_value == 1:
                child.decision = -1 if is_0_positive else 1

    def predict_dataset(self, data):
        predictions = []

        for index, row in data.iterrows():
            prediction = self.predict(self.root, row)
            predictions.append(prediction)

        return predictions

    def predict(self, node, data_point):
        if node.decision is not None:
            return node.decision

        attribute = node.attribute
        matching_child = None
        for child in node.children:
            if child.parent_attribute_value == data_point[attribute]:
                matching_child = child
                break

        return self.predict(matching_child, data_point)


def build_hypotheses(data):
    classifiers = []

    all_positive = Tree(data, None, None, True)
    all_negative = Tree(data, None, None, False)
    classifiers.extend([all_negative, all_positive])

    for attribute in data.columns:
        if attribute not in ['OVERALL_DIAGNOSIS', 'W']:
            true_classifier = Tree(data, attribute, True, None)
            classifiers.append(true_classifier)

            false_classifier = Tree(data, attribute, False, None)
            classifiers.append(false_classifier)

    for classifier in classifiers:
        classifier.classify()

    return classifiers


def update_alphas(alphas, classifiers, data):
    num_classifiers = len(alphas)
    current_index = 0
    has_completed = False

    while not has_completed:
        current_alpha = alphas[current_index]
        num_samples, num_features = data.shape

        labels = data['OVERALL_DIAGNOSIS'].values.reshape(num_samples, 1)
        predicted_labels = np.array(classifiers[current_index].predict_dataset(data)).reshape(num_samples, 1)

        misclassification_mask = abs(labels - predicted_labels)

        misclassified_indices = np.where(misclassification_mask > 0)[0]
        correctly_classified_indices = np.where(misclassification_mask <= 0)[0]

        misclassified_data = data.iloc[misclassified_indices]
        correctly_classified_data = data.iloc[correctly_classified_indices]

        num_misclassified_samples = len(misclassified_indices)
        num_correctly_classified_samples = len(correctly_classified_indices)

        labels_misclassified = data['OVERALL_DIAGNOSIS'].iloc[misclassified_indices].values.reshape(
            (num_misclassified_samples, 1))
        labels_correctly_classified = data['OVERALL_DIAGNOSIS'].iloc[correctly_classified_indices].values.reshape(
            (num_correctly_classified_samples, 1))

        predicted_labels_misclassified = np.zeros((num_misclassified_samples, 1))
        predicted_labels_correctly_classified = np.zeros((num_correctly_classified_samples, 1))

        for i in range(len(alphas)):
            if i == current_index:
                continue

            alpha = alphas[i]
            predicted_labels_misclassified += alpha * (
                np.array(classifiers[i].predict_dataset(misclassified_data)).reshape(num_misclassified_samples, 1))
            predicted_labels_correctly_classified += alpha * (
                np.array(classifiers[i].predict_dataset(correctly_classified_data)).reshape(
                    num_correctly_classified_samples, 1))

        correctly_classified_sum = np.exp(-1.0 * labels_correctly_classified * predicted_labels_correctly_classified).sum()
        misclassified_sum = np.exp(-1.0 * labels_misclassified * predicted_labels_misclassified).sum()

        updated_alpha = 0.5 * np.log(correctly_classified_sum / misclassified_sum)

        difference = updated_alpha - current_alpha

        if abs(difference) > 1e-4:
            alphas[current_index % num_classifiers] = updated_alpha
            return alphas, current_index % num_classifiers

        if current_index > num_classifiers:
            has_completed = True
            return alphas, None

        current_index += 1


def coordinate_descent(data, classifiers):
    num_classifiers = len(classifiers)
    alphas = np.zeros(num_classifiers).reshape(num_classifiers, 1)

    iteration_count = 0
    has_reached_optimum = False

    while not has_reached_optimum:
        alphas, counter = update_alphas(alphas, classifiers, data)

        iteration_count += 1
        if counter is None:
            has_reached_optimum = True

    print('Optimal Alphas:', alphas.flatten())
    return alphas


def calculate_accuracy(data, alphas, classifiers):
    num_samples, num_rows = data.shape
    true_labels = data['OVERALL_DIAGNOSIS'].values.reshape(num_samples, 1)
    predicted_labels = np.zeros((num_samples, 1))

    num_classifiers = len(classifiers)

    for i in range(num_classifiers):
        alpha = alphas[i]
        predicted_labels += alpha * np.array(classifiers[i].predict_dataset(data)).reshape(num_samples, 1)

    predicted_labels = np.sign(predicted_labels)
    misclassifications = np.sum(predicted_labels != true_labels) / 2

    accuracy = (1.0 - misclassifications / num_samples) * 100
    return accuracy


classifiers = build_hypotheses(training_data)
alphas = coordinate_descent(training_data, classifiers)

print('Training Accuracy: ', calculate_accuracy(training_data, alphas, classifiers))
print('Test Accuracy: ', calculate_accuracy(test_data, alphas, classifiers))
