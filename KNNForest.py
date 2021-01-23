from ID3 import ID3, Tree
import math
import numpy as np
import pandas as pd
from scipy import spatial, stats


class KNNDecisionTree(ID3):
    def __init__(self, N, K, P=0.35):
        ID3.__init__(self)  # initializing train and test CSVs.
        # self.normalized_train_samples, _ = KNNDecisionTree.process_csv('train')
        # self.normalized_train_samples = self.train_samples
        self.test_features_values, self.test_target_diagnosis = KNNDecisionTree.process_csv('test')
        # self.normalize_data()
        self.size_of_train_group = self.train_samples.shape[0] - 1  # number of rows in train.csv - 1 (for features)
        self.N = N  # number of decision trees for fit.
        self.K = K  # number of sub-trees to choose.
        self.probability = P
        self.classifiers = [None] * self.N  # N decision trees for 'fit'
        self.centroids = np.zeros(shape=(self.N, self.train_samples.shape[1] - 1), dtype=float),  # matrix of features.

    """
    def normalize_data(self):
        tmp_min = self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]].min()
        tmp_max = self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]].max()
        # normalizing train samples by MinMax
        self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]] -= \
            self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]].min()
        self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]] /= \
            self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]].max()
        # normalizing test samples by MinMax, where min and max are from train samples.
        self.test_features_values -= tmp_min
        self.test_features_values /= tmp_max
    """

    @staticmethod
    def process_csv(csv):
        data = pd.read_csv(f'{csv}.csv')
        features = data[data.columns.values[1:]]
        diagnosis = data[data.columns.values[0]]
        features = features.apply(pd.to_numeric, errors='coerce')
        # diagnosis = diagnosis.apply(pd.to_numeric, errors='coerce') # todo: perhaps need to change B, M to 0 and 1
        return features, diagnosis

    def _predict_single(self, sample_to_classify):
        """
        This method classifies a single sample according to the K-nearest centroids.
        :param sample_to_classify: the samples to classify
        :return: a prediction of 'M' or 'B'.
        """
        distances = [(spatial.distance.euclidean(sample_to_classify, centroid), idx)  # (distance, index)
                     for idx, centroid in enumerate(self.centroids[0])]
        # distances is an array of the euclidean distances from the centroids
        k_nearest = sorted(distances)[:self.K]
        k_trees_diagnosis = []
        for _, ind in k_nearest:
            k_trees_diagnosis.append(KNNDecisionTree._classify(sample_to_classify, self.classifiers[ind]))
        prediction = max(k_trees_diagnosis, key=k_trees_diagnosis.count)  # we classify according to what we have more.
        return prediction

    def predict(self):
        predictions = []
        for index, predictors in self.test_features_values.iterrows():
            # we iterate over each sample in the test, and classify it.
            predictions.append(self._predict_single(predictors))  # list of classifications
        return predictions

    def fit(self):
        # we study N ID3 decision trees
        samples_to_choose = math.ceil(self.probability * self.size_of_train_group)
        for tree in range(self.N):
            # first we randomly choose p*n samples from the train-set.
            # ith_train_samples = self.train_samples.sample(n=samples_to_choose)
            ith_train_samples = self.train_samples.sample(n=samples_to_choose)
            self.centroids[0][tree] = ith_train_samples.mean(axis=0)  # calculating the centroid of the tree
            self.classifiers[tree] = self._get_classifier_tree(self.features_names, ith_train_samples)


if __name__ == '__main__':
    knn_tree = KNNDecisionTree(N=10, K=7)
    knn_tree.fit()
    res_predictions = knn_tree.predict()
    accuracy = knn_tree.accuracy(res_predictions)
    print(accuracy)
