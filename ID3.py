import pandas as pd
from numpy import log2
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from time import time
import numpy as np


class Tree:
    """
    This class represents a classifier tree to be used in the algorithms.
    """

    def __init__(self, feature, threshold, children, classification=None):
        self.feature = feature  # The feature to split by
        self.threshold = threshold
        self.children = children  # Decision tree branches.
        self.classification = classification


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
        :param b_samples: the number of b samples
        :param m_samples: the number of m samples
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
    def _calculate_feature_ig(feature, values, m_diagnosis_numbered, b_diagnosis_numbered, side_prob, entropy):
        f_values = values[feature].to_numpy()
        f_idxs = f_values.argsort()
        sorted_diagnosis = m_diagnosis_numbered[f_idxs][:, np.newaxis].repeat(len(m_diagnosis_numbered), 1)
        left_num_of_m = np.triu(sorted_diagnosis).sum(axis=0)
        right_num_of_m = np.tril(sorted_diagnosis).sum(axis=0)
        sorted_diagnosis = b_diagnosis_numbered[f_idxs][:, np.newaxis].repeat(len(b_diagnosis_numbered), 1)
        left_num_of_b = np.triu(sorted_diagnosis).sum(axis=0)
        right_num_of_b = np.tril(sorted_diagnosis).sum(axis=0)
        left_prob_of_m = left_num_of_m / (left_num_of_m + left_num_of_b)
        right_prob_of_m = right_num_of_m / (right_num_of_m + right_num_of_b)
        left_entropy = ID3._entropy(left_prob_of_m)
        right_entropy = ID3._entropy(right_prob_of_m)
        info_gain = entropy - left_entropy * side_prob - right_entropy * (1 - side_prob)
        return info_gain, f_values, f_idxs

    @staticmethod
    def _entropy(probs):
        def log2_(x):
            z = np.zeros_like(x)
            z[x != 0] = log2(x[x != 0])
            return z

        entropy = -probs * log2_(probs) - (1 - probs) * log2_(1 - probs)
        entropy[probs == 0] = 0
        return entropy

    @staticmethod
    def _max_feature(features, values: pd.DataFrame):
        """
        Finds the feature that maximizes the information-gain, and the best threshold for that feature.
        :param features: list, List containing the feature IDs (from the .csv).
        :param values: list, List containing the values of each subject and feature (from the .csv).
        :returns: string and int, feature and threshold.
        """
        b_samples, m_samples = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        m_diagnosis_numbered = (values['diagnosis'] == "M").to_numpy() * 1.0
        b_diagnosis_numbered = 1 - m_diagnosis_numbered
        entropy = ID3._get_entropy(b_samples.shape[0], m_samples.shape[0])
        best_feature = None
        best_info_gain = float('-inf')
        best_feature_threshold = 0
        side_prob = (1 + np.arange(len(m_diagnosis_numbered))) / len(m_diagnosis_numbered)
        for feature in features:
            info_gain, f_values, f_idxs = ID3._calculate_feature_ig(feature, values, m_diagnosis_numbered,
                                                                    b_diagnosis_numbered, side_prob, entropy)
            max_feat_info_gain = info_gain.max()
            max_idx = info_gain.argmax()
            threshold = f_values[f_idxs][max_idx:max_idx + 2].mean()
            if best_info_gain <= max_feat_info_gain:
                best_info_gain = max_feat_info_gain
                best_feature = feature
                best_feature_threshold = threshold

        return best_feature, best_feature_threshold

    def _get_classifier_tree(self, samples, min_samples=1, features=None):
        """
        :param min_samples:
        :param features: the keys from the .csv (first row).
        :param samples: the keys' samples from the .csv (from the second row and on).
        :rtype: classifierTree, a classifying tree based on the features and samples.
        """
        if features is None:
            features = self.features_names
        values = samples.shape[0]
        m_samples = samples.loc[samples['diagnosis'] == 'M']
        number_m_samples = m_samples.shape[0]
        default = 'B' if (values - number_m_samples) >= (values / 2) else 'M'
        n_default_samples = values - number_m_samples if default == "B" else number_m_samples

        if n_default_samples == values \
                or features.shape[0] == 0 \
                or number_m_samples < min_samples:
            return Tree(None, None, None, default)

        best_feature, threshold = self._max_feature(features, samples)
        left = samples.loc[samples[best_feature] < threshold]
        right = samples.loc[samples[best_feature] >= threshold]  # if == threshold, we go right
        if len(left) == 0 or len(right) == 0:
            return Tree(None, None, None, default)

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
        else:
            targets = targets[targets.columns.values[0]]
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
        m_values = [0, 1]
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
            # print(f"ended {val} M value. Avg accuracy is: {avg_acc_per_m}")
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
    id3_alg.fit()
    # IN ORDER TO SEE THE EXPERIMENT VALUES, TAKE DOWN THE BOTH FOLLOWING COMMENTS. IF A GRAPH IS NOT NEEDED, SEND False
    # id3_alg.experiment(predict_or_loss='loss', print_graph=True)
    # exit(0)
    res_predictions = id3_alg.predict()
    accuracy = id3_alg.accuracy(res_predictions)
    print(accuracy)
    # print(id3_alg.loss(res_predictions))
