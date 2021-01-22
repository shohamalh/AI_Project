from ID3 import ID3, Tree
import math
import numpy as np
import pandas as pd
from scipy import spatial


class KNNDecisionTree(ID3):
    def __init__(self, N, K, p):
        ID3.__init__(self)  # initializing train and test CSVs.
        # self.normalized_train_samples, _ = KNNDecisionTree.process_csv('train')
        self.normalized_train_samples = self.train_samples
        self.test_features_values, self.test_target_diagnosis = KNNDecisionTree.process_csv('test')
        self.normalize_data()
        self.size_of_train_group = self.train_samples.shape[0] - 1  # number of rows in train.csv - 1 (for features)
        self.N = N  # number of decision trees for fit.
        self.K = K  # number of sub-trees to choose.
        self.probability = p  # todo: will be determined later using experiments.
        self.classifiers = [None] * self.N  # N decision trees for 'fit'
        self.centroids = np.zeros(shape=(self.N, self.train_samples.shape[1] - 1), dtype=float),  # matrix of features.

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
        # given a vector of values, we want to classify it according to the K closest centroids.
        # we normalize with MinMax.
        """
        given a sample and the K nearest trees, we classify as the max of these trees.
        """
        distances = [(spatial.distance.euclidean(sample_to_classify, centroid), idx)  # (distance, index)
                     for idx, centroid in enumerate(self.centroids[0])]
        # distances is an array of the euclidean distances from the centroids
        k_nearest = sorted(distances)[:self.K]
        k_trees_diagnosis = []
        for d, i in k_nearest:
            k_trees_diagnosis.append(KNNDecisionTree.classify(sample_to_classify, self.classifiers[i]))
        prediction = max(k_trees_diagnosis, key=k_trees_diagnosis.count)  # we classify according to what we have more.
        return prediction

    def predict(self):
        predictions = []
        for index, predictors in self.test_features_values.iterrows():
            # we iterate over each sample in the test, and classify it.
            predictions.append(self._predict_single(predictors))  # list of classifications
        return predictions

    def accuracy(self, predictions):
        correct_predictions = 0
        for i in range(len(predictions)):
            if self.test_target_diagnosis[i] == predictions[i]:
                correct_predictions += 1
        return correct_predictions / len(predictions)

    @staticmethod
    def classify(sample_to_classify, classifier: Tree):
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

    def fit(self):
        # we study N ID3 decision trees
        samples_to_choose = math.ceil(self.probability * self.size_of_train_group)
        for tree in range(self.N):
            # first we randomly choose p*n samples from the train-set.
            # ith_train_samples = self.train_samples.sample(n=samples_to_choose)
            ith_train_samples = self.normalized_train_samples.sample(n=samples_to_choose)
            self.classifiers[tree] = self.get_classifier_tree(self.features_names, ith_train_samples)
            self.centroids[0][tree] = ith_train_samples.mean(axis=0)  # calculating the centroid of the tree


if __name__ == '__main__':
    # K =
    # N =
    best_p = 0.3
    highest_acc = 0
    highest_avg_acc = 0
    for i in range(35):
        avg_acc = 0
        p = 0.3 + i * 0.02
        knn_tree = KNNDecisionTree(N=10, K=5, p=p)
        for j in range(10):
            knn_tree.fit()
            res_predictions = knn_tree.predict()
            accuracy = knn_tree.accuracy(res_predictions)
            print(j + 1, 'th iteration accuracy:', accuracy, 'probability ', p)
            if accuracy >= highest_acc:
                highest_acc = accuracy
            avg_acc += accuracy
        if avg_acc >= highest_avg_acc:
            highest_avg_acc = avg_acc
            best_p = p
        print('average accuracy of p =', p, ' is', avg_acc)
    print('highest accuracy:', highest_acc, ' p:', best_p)
