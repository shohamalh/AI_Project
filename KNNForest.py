from ID3 import ID3, Tree
import math
import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from time import time


class KNNDecisionTree(ID3):
    def __init__(self, N, K, P=0.35):
        ID3.__init__(self)  # initializing train and test CSVs.
        self.test_features_values = self.test_samples[self.test_samples.columns.values[1:]]
        self.size_of_train_group = self.train_samples.shape[0] - 1  # number of rows in train.csv - 1 (for features)
        self.N = N  # number of decision trees for fit.
        self.K = K  # number of sub-trees to choose.
        self.probability = P
        self.classifiers = [None] * self.N  # N decision trees for 'fit'
        self.centroids = np.zeros(shape=(self.N, self.train_samples.shape[1] - 1), dtype=float),  # matrix of features.

    def _calculate_distances(self):
        return spatial.distance.cdist(self.test_samples[self.test_samples.columns.values[1:]].to_numpy(),
                                      self.centroids[0])

    def _predict_single(self, sample_to_classify, sorted_centroid_idxs):
        """
        This method classifies a single sample according to the K-nearest centroids.
        :param sample_to_classify: the samples to classify
        :return: a prediction of 'M' or 'B'.
        """
        k_trees_diagnosis = []
        for ind in sorted_centroid_idxs[:self.K]:
            k_trees_diagnosis.append(KNNDecisionTree._classify(sample_to_classify, self.classifiers[ind]))
        prediction = max(k_trees_diagnosis, key=k_trees_diagnosis.count)  # we classify according to the Majority Class.
        return prediction

    def predict(self, test_set=None):
        predictions = []
        if test_set is None:
            test_set = self.test_features_values
        else:
            test_set = test_set[test_set.columns.values[1:]]
        dists = self._calculate_distances()  # distances matrix
        sorted_idxs_by_dist = dists.argsort(axis=1)
        for (_, sample), sorted_idxs_for_sample in zip(test_set.iterrows(), sorted_idxs_by_dist):
            # we iterate over each sample in the test, and classify it.
            print(self)
            predictions.append(self._predict_single(sample, sorted_idxs_for_sample))  # list of classifications
        return predictions

    def fit(self, train_samples=None, size_of_train_group=0):
        if train_samples is None:
            train_samples = self.train_samples
        if 0 == size_of_train_group:
            size_of_train_group = self.size_of_train_group
        # we study N ID3 decision trees
        samples_to_choose = math.ceil(self.probability * size_of_train_group)
        for tree in range(self.N):
            # first we randomly choose p*n samples from the train-set.
            ith_train_samples = train_samples.sample(n=samples_to_choose)
            self.centroids[0][tree] = ith_train_samples.mean(axis=0)  # calculating the centroid of the tree
            self.classifiers[tree] = self._get_classifier_tree(ith_train_samples)


if __name__ == '__main__':
    t = time()
    knn_tree = KNNDecisionTree(N=10, K=7)
    knn_tree.fit()
    res_predictions = knn_tree.predict()
    accuracy = knn_tree.accuracy(res_predictions)
    print(accuracy)
    print(time() - t)
