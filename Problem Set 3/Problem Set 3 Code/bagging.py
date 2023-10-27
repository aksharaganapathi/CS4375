import pandas as pd
import numpy as np

column_names = ['OVERALL_DIAGNOSIS'] + [f'F{i}' for i in range(1, 23)]
training_data = pd.read_csv('heart_train.data', names=column_names, header=None)
test_data = pd.read_csv('heart_test.data', names=column_names, header=None)


class Node(object):

    def __init__(self, data, parent_attribute_value):
        self.data = data
        self.children = []
        self.attribute = None
        self.decision = None
        self.parent_attribute = None
        self.parent_attribute_value = parent_attribute_value


class Tree(object):

    def __init__(self, data):
        self.data = data
        self.root = None
        self.training_error = None

    def classify(self):
        node = Node(self.data, None)
        self.root = node
        return self.build_decision_tree(node)

    def find_best_attribute(self, current_node):
        global error_rate_0_positive, error_rate_1_positive
        node_data = current_node.data

        min_error_rate = float('inf')
        best_attribute = None

        for attribute in node_data.columns:
            if attribute in ['OVERALL_DIAGNOSIS', 'W']:
                continue

            error_rate_0_positive = error_rate_1_positive = 0
            unique_attribute_values = node_data[attribute].unique()

            for attribute_value in unique_attribute_values:
                split_data = node_data[node_data[attribute] == attribute_value]
                weighted_sum = split_data.groupby('OVERALL_DIAGNOSIS')['W'].sum()

                if attribute_value == 0:
                    error_rate_0_positive += weighted_sum.get(0, 0)
                    error_rate_1_positive += weighted_sum.get(1, 0)
                elif attribute_value == 1:
                    error_rate_0_positive += weighted_sum.get(1, 0)
                    error_rate_1_positive += weighted_sum.get(0, 0)

            error_rate = min(error_rate_0_positive, error_rate_1_positive)

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_attribute = attribute

        is_zero_positive = error_rate_0_positive < error_rate_1_positive

        return best_attribute, min_error_rate, is_zero_positive

    def build_decision_tree(self, current_node):
        node_data = current_node.data
        best_attribute, error_rate, is_zero_positive = self.find_best_attribute(current_node)
        current_node.attribute = best_attribute

        for attribute_value in node_data[best_attribute].unique():
            split_data = node_data[node_data[best_attribute] == attribute_value]
            child_node = Node(split_data, attribute_value)
            child_node.parent_attribute = best_attribute
            current_node.children.append(child_node)

            if is_zero_positive:
                child_node.decision = 1 - attribute_value
            else:
                child_node.decision = attribute_value

        return error_rate

    def predict_dataset(self, dataset):
        return dataset.apply(lambda data_point: self.predict(self.root, data_point), axis=1).tolist()

    def predict(self, node, data_point):
        if node.decision is not None:
            return node.decision

        attribute = node.attribute
        child_node = next((child for child in node.children if child.parent_attribute_value == data_point[attribute]),
                          None)

        if child_node is not None:
            return self.predict(child_node, data_point)

        return None


def bagging(train_data, num_bootstrap_samples):
    trained_classifiers = []
    num_samples, num_features = train_data.shape
    train_data['W'] = (np.ones(num_samples) * (1.0 / num_samples)).reshape(num_samples, 1)

    for i in range(num_bootstrap_samples):
        bootstrap_sample = train_data.iloc[np.unique(np.random.randint(0, num_samples, size=num_samples))]
        decision_tree = Tree(bootstrap_sample)
        decision_tree.classify()
        trained_classifiers.append(decision_tree)

    return trained_classifiers


def predict_diagnosis(test_data, trained_classifiers):
    num_samples, num_features = test_data.shape
    diagnosis_labels = test_data['OVERALL_DIAGNOSIS'].values.reshape(num_samples, 1)
    predicted_labels = np.zeros(num_samples).reshape(num_samples, 1)

    num_classifiers = len(trained_classifiers)

    for classifier in trained_classifiers:
        predicted_labels += np.array(classifier.predict_dataset(test_data)).reshape(num_samples, 1)

    predicted_labels = (predicted_labels >= (num_classifiers / 2)).astype(int)

    num_misclassifications = np.sum(np.abs(diagnosis_labels - predicted_labels))

    return (1.0 - num_misclassifications / num_samples) * 100


classifiers = bagging(training_data, 20)
print('Training Accuracy: ', predict_diagnosis(training_data, classifiers))
print('Test Accuracy: ', predict_diagnosis(test_data, classifiers))
