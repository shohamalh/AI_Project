import pandas as pd
from numpy import log2
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np


class Tree:
    """
    Contains the information of the node and another nodes of the Decision Tree.
    """

    def __init__(self, feature, threshold, children):
        self.feature = feature  # The feature to split by
        self.threshold = threshold
        self.children = children  # Decision tree branches.
        self.classification = None

    def set_classification(self, classification):
        self.classification = classification


class ID3:
    def __init__(self):
        self.train_samples = pd.read_csv("train.csv")  # for fit
        self.test_samples = pd.read_csv("test.csv")  # for predict
        self.features_names = self.train_samples.keys()[1:]
        self.classifier = None

    @staticmethod
    def get_entropy(positive_samples: int, negative_samples: int) -> float:
        """
        :param positive_samples: the number of positive samples
        :param negative_samples: the number of negative samples
        :return: entropy: float, Entropy.
        """
        total_samples = positive_samples + negative_samples  # total number of samples.
        if total_samples == 0 or 0 in (positive_samples, negative_samples):
            # there are no samples / the samples are pure.
            return 0

        pr_pos = positive_samples / total_samples
        pr_neg = negative_samples / total_samples
        return -pr_pos * log2(pr_pos) - pr_neg * log2(pr_neg)

    @staticmethod
    def max_feature(features, values):
        """
        Finds the feature that maximizes the information-gain, and the best threshold for that feature.
        :param features: list, List containing the feature IDs (from the .csv).
        :param values: list, List containing the values of each subject and feature (from the .csv).
        :returns: string and int, feature and threshold.
        """
        positive, negative = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        entropy = ID3.get_entropy(positive.shape[0], negative.shape[0])
        best_feature = None
        best_info_gain = float('-inf')
        best_feature_threshold = 0
        size = values.shape[0]

        for feature in features:
            sorted_values = sorted(list(values[feature]), key=lambda x: float(x))
            thresholds = [(i + j) / 2 for i, j in zip(sorted_values[:-1], sorted_values[1:])]
            for threshold in thresholds:
                pos_lower = (positive.loc[positive[feature] < threshold]).shape[0]
                neg_lower = (negative.loc[negative[feature] < threshold]).shape[0]
                pos_higher = positive.shape[0] - pos_lower
                neg_higher = negative.shape[0] - neg_lower
                entropy_low = ID3.get_entropy(pos_lower, neg_lower)
                entropy_high = ID3.get_entropy(pos_higher, neg_higher)

                info_gain_feature_lower = (pos_lower + neg_lower) / size * entropy_low
                info_gain_feature_higher = (pos_higher + neg_higher) / size * entropy_high

                info_gain = entropy - info_gain_feature_lower - info_gain_feature_higher
                if best_info_gain < info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_feature_threshold = threshold

        return best_feature, best_feature_threshold

    @staticmethod
    def get_classifier_tree(features, samples, min_samples=1):
        """
        :param min_samples:
        :param features: the keys from the .csv (first row).
        :param samples: the keys' samples from the .csv (from the second row and on).
        :rtype: classifierTree, a classifying tree based on the features and samples.
        """
        values = samples.shape[0]
        m_samples = (samples.loc[samples['diagnosis'] == 'M']).shape[0]
        default = 'B' if (samples.loc[samples['diagnosis'] == 'B']).shape[0] > samples.shape[0] / 2 else 'M'

        if (samples.loc[samples['diagnosis'] == default]).shape[0] == values \
                or features.shape[0] == 0 \
                or m_samples <= min_samples:
            node = Tree(None, None, None)
            node.set_classification(default)
            return node

        best_feature, threshold = ID3.max_feature(features, samples)
        left = samples.loc[samples[best_feature] <= threshold]
        right = samples.loc[samples[best_feature] > threshold]
        children = (ID3.get_classifier_tree(features, left, min_samples),
                    ID3.get_classifier_tree(features, right, min_samples))
        return Tree(best_feature, threshold, children)

    def fit(self, min_samples=1):
        self.classifier = self.get_classifier_tree(self.features_names, self.train_samples, min_samples)
        # return self.classifier

    def predict(self, test_set=None, classifier=None):
        count_success = 0
        if test_set is None:
            test_set = self.test_samples
        if classifier is None:
            base_node = self.classifier
        else:
            base_node = classifier

        for row_num, row in test_set.iterrows():
            node = base_node
            while node.children is not None:
                if row[node.feature] < node.threshold:
                    node = node.children[0]
                else:
                    node = node.children[1]
            if node.classification == row['diagnosis']:
                count_success += 1
        rows = self.test_samples.shape[0]
        return count_success / rows

    # adding loss function here for easier implementation of CostSensitiveID3
    def loss(self, test_set=None, classifier=None):
        count_fp = 0
        count_fn = 0
        if test_set is None:
            test_set = self.test_samples
        if classifier is None:
            base_node = self.classifier
        else:
            base_node = classifier

        for row_num, row in test_set.iterrows():
            node = base_node
            while node.children is not None:
                if row[node.feature] < node.threshold:
                    node = node.children[0]
                else:
                    node = node.children[1]
            if not node.classification == row['diagnosis']:
                if row['diagnosis'] == 'B':
                    count_fp += 1
                else:
                    count_fn += 1

        rows = self.test_samples.shape[0]
        return (count_fp * 0.1 + count_fn) / rows

    def experiment(self, test_or_loss='test'):
        kf = KFold(n_splits=5, shuffle=True, random_state=208501684)
        m_values = np.array((1, 2, 3, 5, 8, 13, 21))
        test_acc = []
        for val in m_values:
            cum_acc = 0
            for fold_id, (train_index, test_index) in enumerate(kf.split(self.train_samples)):
                accuracy = 0
                train_set = self.train_samples.loc[train_index]
                test_set = self.train_samples.loc[test_index]
                id3_tree = self.get_classifier_tree(self.train_samples.keys()[1:], train_set, min_samples=val)
                if test_or_loss == 'test':
                    accuracy = self.test(test_set=test_set, classifier=id3_tree)
                elif test_or_loss == 'loss':
                    accuracy = self.loss(test_set=test_set, classifier=id3_tree)
                print(f"M = {val}, Fold No.: {fold_id} -> Accuracy: {accuracy}")
                cum_acc += accuracy
            avg_acc_per_m = cum_acc / 5
            print(f"ended {min} M value. Avg accuracy is: {avg_acc_per_m}")
            test_acc += [avg_acc_per_m]

        plt.scatter(m_values, test_acc)
        plt.show()


if __name__ == '__main__':
    print("running id3")
    id3_alg = ID3()
    id3_alg.fit()
    # id3_alg.experiment()
    print('the accuracy on the test group is:', id3_alg.predict())
