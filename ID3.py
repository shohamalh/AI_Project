import pandas as pd
from numpy import log2
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class Tree:
    """
    This class represents a classifier tree to be used in the algorithms.
    """

    def __init__(self, feature, threshold, children):
        self.feature = feature  # The feature to split by
        self.threshold = threshold
        self.children = children  # Decision tree branches.
        self.classification = None


class ID3:
    def __init__(self):
        self.train_samples = pd.read_csv("train.csv")  # for fit
        self.test_samples = pd.read_csv("test.csv")  # for predict
        self.test_target_diagnosis = self.test_samples[self.test_samples.columns.values[0]]
        self.features_names = self.train_samples.keys()[1:]
        self.tree = None

    @staticmethod
    def _get_entropy(b_samples: int, m_samples: int) -> float:
        """
        :param b_samples: the number of positive samples
        :param m_samples: the number of negative samples
        :return: entropy: float, Entropy.
        """
        total_samples = b_samples + m_samples  # total number of samples.
        if total_samples == 0 or 0 in (b_samples, m_samples):
            # there are no samples / the samples are pure.
            return 0

        pr_b = b_samples / total_samples
        pr_m = m_samples / total_samples
        return -pr_b * log2(pr_b) - pr_m * log2(pr_m)

    @staticmethod
    def _calculate_feature_ig(feature, threshold, b_samples, m_samples, entropy):
        size = len(b_samples) + len(m_samples)
        b_lower = (b_samples.loc[b_samples[feature] < threshold]).shape[0]
        m_lower = (m_samples.loc[m_samples[feature] < threshold]).shape[0]
        b_higher = b_samples.shape[0] - b_lower
        m_higher = m_samples.shape[0] - m_lower
        entropy_low = ID3._get_entropy(b_lower, m_lower)
        entropy_high = ID3._get_entropy(b_higher, m_higher)
        info_gain_feature_lower = (b_lower + m_lower) / size * entropy_low
        info_gain_feature_higher = (b_higher + m_higher) / size * entropy_high
        return entropy - info_gain_feature_lower - info_gain_feature_higher

    @staticmethod
    def _max_feature(features, values):
        """
        Finds the feature that maximizes the information-gain, and the best threshold for that feature.
        :param features: list, List containing the feature IDs (from the .csv).
        :param values: list, List containing the values of each subject and feature (from the .csv).
        :returns: string and int, feature and threshold.
        """
        b_samples, m_samples = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        entropy = ID3._get_entropy(b_samples.shape[0], m_samples.shape[0])
        best_feature = None
        best_info_gain = float('-inf')
        best_feature_threshold = 0
        for feature in features:
            sorted_values = sorted(list(values[feature]), key=lambda x: float(x))
            thresholds = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
            for threshold in thresholds:
                info_gain = ID3._calculate_feature_ig(feature, threshold, b_samples, m_samples, entropy)
                if best_info_gain <= info_gain:
                    best_info_gain = max(best_info_gain, info_gain)
                    best_feature = feature
                    best_feature_threshold = threshold

        return best_feature, best_feature_threshold

    def _get_classifier_tree(self, samples, min_samples=1):
        """
        :param min_samples:
        :param features: the keys from the .csv (first row).
        :param samples: the keys' samples from the .csv (from the second row and on).
        :rtype: classifierTree, a classifying tree based on the features and samples.
        """
        features = self.features_names
        values = samples.shape[0]
        m_samples = (samples.loc[samples['diagnosis'] == 'M']).shape[0]
        default = 'B' if (samples.loc[samples['diagnosis'] == 'B']).shape[0] >= samples.shape[0] / 2 else 'M'

        if (samples.loc[samples['diagnosis'] == default]).shape[0] == values \
                or features.shape[0] == 0 \
                or m_samples < min_samples:
            node = Tree(None, None, None)
            node.classification = default
            return node

        best_feature, threshold = self._max_feature(features, samples)
        left = samples.loc[samples[best_feature] < threshold]
        right = samples.loc[samples[best_feature] >= threshold]  # if == threshold, we go right
        children = (self._get_classifier_tree(left, min_samples),
                    self._get_classifier_tree(right, min_samples))
        return Tree(best_feature, threshold, children)

    @staticmethod
    def _classify(sample_to_classify, classifier: Tree):
        """
        this method returns the classification of the tree.
        :param sample_to_classify:
        :param classifier:
        :return:
        """
        while classifier.children is not None:
            if sample_to_classify[classifier.feature] < classifier.threshold:
                classifier = classifier.children[0]
            else:
                classifier = classifier.children[1]
        return classifier.classification

    def fit(self, min_samples=1):
        self.tree = self._get_classifier_tree(self.train_samples, min_samples)

    def predict(self, test_set=None, classifier=None):
        predicting_tree = self.tree
        if test_set is None:
            test_set = self.test_samples
        if classifier is not None:
            predicting_tree = classifier
        predictions = []
        for _, predictors in test_set.iterrows():
            predictions.append(self._classify(predictors, predicting_tree))
        return predictions

    def loss(self, predictions, targets=None):
        count_fp = 0
        count_fn = 0
        if targets is None:
            targets = self.test_target_diagnosis
        for i in range(len(predictions)):
            if not (targets.to_numpy())[i] == predictions[i]:
                if predictions[i] == 'B':
                    count_fn += 1
                else:
                    count_fp += 1
        return (count_fp * 0.1 + count_fn) / len(predictions)

    def accuracy(self, predictions, targets=None):
        if targets is None:
            targets = self.test_target_diagnosis
        correct_predictions = 0
        for i in range(len(predictions)):
            if (targets.to_numpy())[i] == predictions[i]:
                correct_predictions += 1
        return correct_predictions / len(predictions)

    def experiment(self, predict_or_loss='predict', print_graph=False):
        """
        NOTE: In order to print the graph, call with print_graph=True
        :param predict_or_loss: what we are checking the predict or loss
        :param print_graph:
        :return:
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=208501684)
        m_values = [1, 5, 10, 20, 50, 100]
        test_acc = []
        for val in m_values:
            cum_acc = 0
            for fold_id, (train_index, test_index) in enumerate(kf.split(self.train_samples)):
                acc = 0
                train_samples = self.train_samples.loc[train_index]
                test_samples = self.train_samples.loc[test_index]
                id3_tree = self._get_classifier_tree(train_samples, min_samples=val)
                if predict_or_loss == 'predict':
                    acc = self.accuracy(self.predict(test_set=test_samples, classifier=id3_tree),
                                        test_samples['diagnosis'])
                elif predict_or_loss == 'loss':
                    acc = self.loss(self.predict(test_set=test_samples, classifier=id3_tree),
                                    test_samples['diagnosis'])
                # print(f"M = {val}, Fold No.: {fold_id} -> Accuracy: {acc}")
                cum_acc += acc
            avg_acc_per_m = cum_acc / 5
            print(f"ended {val} M value. Avg accuracy is: {avg_acc_per_m}")
            test_acc.append(avg_acc_per_m)
        if print_graph:
            plt.scatter(m_values, test_acc)
            plt.xlabel('M')
            if predict_or_loss == 'loss':
                plt.ylabel('Loss')
            else:
                plt.ylabel('Accuracy')
            plt.show()


if __name__ == '__main__':
    id3_alg = ID3()
    id3_alg.fit(1)
    # id3_alg.experiment(predict_or_loss='predict', print_graph=True)
    res_predictions = id3_alg.predict()
    accuracy = id3_alg.loss(res_predictions)
    print(accuracy)
    # print(id3_alg.loss(res_predictions))
