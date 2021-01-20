from ID3 import ID3, Tree
import math
import numpy as np
import pandas as pd
from scipy import spatial
import heapq


class KNNDecisionTree(ID3):
    # predict = on test group.
    # fit = study
    def __init__(self, N, K):
        # ID3.__init__(self)  # initializing train and test CSVs.
        self.train_samples = pd.read_csv("train.csv")  # for fit
        self.test_samples = pd.read_csv("test.csv")  # for predict
        self.features_names = self.train_samples.keys()[1:]

        predictors, targets = KNNDecisionTree.get_data('train')
        self.predictors_data, self.predictors_targets = KNNDecisionTree.get_data('test')
        self._train_min = predictors.min().to_numpy()
        self._train_max = predictors.max().to_numpy()
        self.normalized_train = (predictors - predictors.min()) / (predictors.max() - predictors.min())
        self.normalized_train = self.normalized_train.values

        self.size_of_train_group = self.train_samples.shape[0] - 1  # number of rows in train.csv - 1 (for header)
        self.N = N  # number of decision trees for fit.
        self.K = K  # number of sub-trees to choose.

        self.probability = 0.4  # will be determined later using experiments. todo:
        self.classifiers = [None] * self.N  # N decision trees for 'fit'
        self.number_of_features = self.train_samples.shape[1] - 1
        self.centroids = np.zeros(shape=(self.N, self.train_samples.shape[1] - 1), dtype=float)  # matrix of features.

        """
        each one of the N trees will have a size-31 vector of the average features,
        where each cell will be the average of the n*p samples chosen.
        """

    @staticmethod
    def process_csv(csv):
        data = pd.read_csv(f'{csv}.csv')
        features = data[data.columns.values[1:]]
        results = data[data.columns.values[0]]
        features = features.apply(pd.to_numeric, errors='coerce')
        results = results.apply(pd.to_numeric, errors='coerce')
        return features, results

    def _predict_single(self, features, K):
        features = (features - self._train_min) / (self._train_max - self._train_min)
        distances = [(spatial.distance.euclidean(features, self_train), self_train_target)
                     for self_train, self_train_target in zip(self._final_train, self._train_targets)]
        # distances is an array of the euclidean distances from the centroids
        nearest = heapq.nsmallest(K, distances, key=lambda x: x[0])  # we take the K smallest (=closest) samples from distances
        nearest_targets = [e[1] for e in nearest]
        target = max(set(nearest_targets), key=nearest_targets.count)
        return target

    def predict(self, K):
        predictions = []
        for predictors, index in zip(self.predictors_data.values, range(self.predictors_data.shape[0])):
            prediction = self._predict_single(predictors, K)
            good_one = 1 if prediction == self.predictors_targets.iloc[[index]].values[0] else 0
            predictions.append(tuple(prediction, good_one))
        accuracy = list(filter(lambda x: x[1] == 1, predictions))
        print(str(len(accuracy) / self.predictors_data.shape[0]))

    @staticmethod
    def get_classifier_tree(features, samples, min_samples=1):
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

    def fit(self, min_samples=1):  # wrapper function
        # we study N ID3 decision trees
        samples_to_choose = math.ceil(self.probability * self.size_of_train_group)
        for tree in range(self.N):
            # first we randomly choose p*n samples from the train-set.
            ith_train_samples = self.train_samples.sample(n=samples_to_choose)
            self.centroids[tree] = ith_train_samples.mean(axis=0)  # calculating the centroid of the tree
            self.classifiers[tree] = self.get_classifier_tree(self.features_names, ith_train_samples, min_samples)

    # def fit(self, X, Y):
    #   raise NotImplementedError


if __name__ == '__main__':
    # K =
    # N =
    knn = KNNDecisionTree(K=5, N=10)
    knn.fit()
    knn.predict()
    print("after fit")
