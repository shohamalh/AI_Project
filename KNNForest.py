from ID3 import ID3
import math
import numpy as np
from scipy import spatial
from time import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class KNNDecisionTree(ID3):
    def __init__(self, N=65, K=7, P=0.6555555555555554):
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


def knn_experiment(print_graph=False):
    best_avg_acc = 0
    best_N = 0
    best_K = 0
    best_p = 0
    highest_acc = 0
    p_vals = np.linspace(0.3, 0.7, 10)
    ks = [5, 7, 9, 11, 13]
    ns = np.linspace(max(ks), max(ks) * 10, 10, dtype=int)
    for K in ks:  # we check every even N from 10 to 30, just so it won't run for ages.
        for N in ns:
            test_acc = []
            for p in p_vals:  # we also try to check the p parameter
                avg_acc = 0
                cum_acc = 0
                experiment_knn = KNNDecisionTree(N=N, K=K, P=p)
                kf = KFold(n_splits=5, shuffle=True, random_state=208501684)
                for fold_id, (train_index, test_index) in enumerate(kf.split(experiment_knn.train_samples)):
                    acc = 0
                    train_samples = experiment_knn.train_samples.loc[train_index]
                    test_samples = experiment_knn.train_samples.loc[test_index]
                    experiment_knn.fit(train_samples, train_samples.shape[0] - 1)  # creating trees
                    predictions = experiment_knn.predict(test_samples)
                    acc = experiment_knn.accuracy(predictions, test_samples)
                    # print(f'N = {N}, K = {K}, p = {p}, Fold = {fold_id} Accuracy = {acc}')
                    if acc >= highest_acc:
                        highest_acc = acc
                    cum_acc += acc
                avg_acc = cum_acc / 5
                # print(f'N = {N}, K = {K}, p = {p}, avg_acc = {avg_acc}')
                test_acc.append(avg_acc)
                if avg_acc >= best_avg_acc:
                    best_avg_acc = avg_acc
                    best_N = N
                    best_K = K
                    best_p = p
            if print_graph:
                plt.scatter(p_vals, test_acc)
                plt.xlabel('p')
                plt.ylabel('Accuracy')
                plt.title(f'N = {N}, K = {K}')
                plt.show()

    print(f'Best values are: N = {best_N}, K = {best_K}, p = {best_p} with avg_acc = {best_avg_acc}')
    print(f'highest acc is: {highest_acc}')
    return best_N, best_K, best_p


if __name__ == '__main__':
    # N, K, P = knn_experiment(False) = 66, 7, 0.6555555555555554
    knn_tree = KNNDecisionTree()
    knn_tree.fit()
    res_predictions = knn_tree.predict()
    accuracy = knn_tree.accuracy(res_predictions)
    print(accuracy)
