from math import log, e
import pandas as pd
from numpy import log2

class Node:
    def __init__(self, feature, th, children):  # TODO: change names.
        self.feature = feature  # The feature to split by
        self.th = th
        self.children = children  # Desicition tree branches.
        self.classification = None

    def set_classification(self, classification):
        self.classification = classification


class ID3Algorithm:
    def __init__(self):
        self.train_samples = pd.read_csv("train.csv")
        self.test_samples = pd.read_csv("test.csv")
        self.classifier = None

        self.features_names = self.train_samples.keys()[1:]

    # send the number of positive and negative
    @staticmethod
    def calculate_entropy(pos_num: int, neg_num: int) -> float:
        """

        :param pos_num: number of positive samples
        :param neg_num: number of negative samples
        :return: Entropy value of dataset
        """
        total_size = pos_num+neg_num
        if total_size == 0:
            return 0
        if 0 in (pos_num, neg_num):
            return 0
        else:
            probability_pos = pos_num/total_size
            probability_neg = neg_num/total_size
            entropy = -probability_pos * log2(probability_pos)
            entropy += -probability_neg * log2(probability_neg)
            return entropy

    # chooses the best feature and returns the feature and the best treshold for this feature
    @staticmethod
    def max_feature(features, values):
        positive, negative = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        entropy = ID3Algorithm.calculate_entropy(positive.shape[0], negative.shape[0])
        best_feature = None
        best_IG = float('-inf')
        best_feature_th = 0
        size = values.shape[0]
        for feature in features:
            sorted_values = sorted(list(values[feature]), key=lambda x: float(x))
            tresholds = [(i + j) / 2 for i, j in zip(sorted_values[:-1], sorted_values[1:])]
            for th in tresholds:
                pos_lower = (positive.loc[positive[feature] < th]).shape[0]
                neg_lower = (negative.loc[negative[feature] < th]).shape[0]
                pos_higher = positive.shape[0] - pos_lower
                neg_higher = negative.shape[0] - neg_lower
                entropy_low = ID3Algorithm.calculate_entropy(pos_lower, neg_lower)
                entropy_high = ID3Algorithm.calculate_entropy(pos_higher, neg_higher)
                ig = entropy - (pos_lower + neg_lower) / size * entropy_low - (
                        pos_higher + neg_higher) / size * entropy_high
                best_IG = max(best_IG, ig)
                if (ig == best_IG):
                    best_feature = feature
                    best_feature_th = th

        return best_feature, best_feature_th

    @staticmethod
    def get_classifier_tree(features, values):
        samples = values.shape[0]
        default = 'B' if (values.loc[values['diagnosis'] == 'B']).shape[0] >= values.shape[0] / 2 else 'M'
        if (values.loc[values['diagnosis'] == default]).shape[0] == samples or features.shape[0] == 0:
            node = Node(None, None, None)
            node.set_classification(default)
            return node

        best_feature, th = ID3Algorithm.max_feature(features, values)
        left = values.loc[values[best_feature] < th]
        right = values.loc[values[best_feature] >= th]
        childern = (ID3Algorithm.get_classifier_tree(features, left), ID3Algorithm.get_classifier_tree(features, right))
        return Node(best_feature, th, childern)

    def train(self):
        self.classifier = self.get_classifier_tree(self.train_samples.keys()[1:], self.train_samples)

    def test(self):
        count_error = 0
        for row_num, row in self.test_samples.iterrows():
            node = self.classifier
            while not node.children == None:
                if row[node.feature] < node.th:
                    node = node.children[0]
                else:
                    node = node.children[1]
            if not node.classification == row['diagnosis']:
                count_error += 1
        rows = self.test_samples.shape[0]
        return count_error / rows


if __name__ == '__main__':
    id3 = ID3Algorithm()
    id3.train()
    print(str(id3.test() * 100))
