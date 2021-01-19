from ID3 import ID3, Tree
import math


class KNNDecisionTree(ID3):
    # predict = on test group.
    # fit = study
    def __init__(self, N, K):
        ID3.__init__(self)  # initializing train and test CSVs.
        self.size_of_train_group = self.train_samples.shape[0] - 1  # number of rows in train.csv - 1 (for header)
        self.N = N  # number of decision trees for fit.
        self.K = K  # number of sub-trees to choose.
        self.probability = 0.4  # will be determined later using experiments.
        self.classifiers = [None] * self.N  # N decision trees for 'fit'
        self.centroids = [0] * self.N
        """
        each one of the N trees will have a size-31 vector of the average features,
        where each cell will be the average of the n*p samples chosen.
        """

    def predict(self, k):
        return

    @staticmethod
    def get_classifier_tree(features, values, min_samples=1):
        samples = values.shape[0]
        m_samples = (values.loc[values['diagnosis'] == 'M']).shape[0]
        default = 'B' if (values.loc[values['diagnosis'] == 'B']).shape[0] > values.shape[0] / 2 else 'M'
        if (values.loc[values['diagnosis'] == default]).shape[0] == samples \
                or features.shape[0] == 0 \
                or m_samples <= min_samples:
            node = Tree(None, None, None)
            node.set_classification(default)
            return node

        best_feature, threshold = ID3.max_feature(features, values)
        left = values.loc[values[best_feature] <= threshold]
        right = values.loc[values[best_feature] > threshold]
        children = (ID3.get_classifier_tree(features, left, min_samples),
                    ID3.get_classifier_tree(features, right, min_samples))
        return Tree(best_feature, threshold, children)

    def fit(self, min_samples=1):  # wrapper function
        # we study N ID3 decision trees
        samples_to_choose = math.ceil(self.probability * self.size_of_train_group)
        for i in range(self.N):
            # first we randomly choose p*n samples from the train-set.
            ith_train_samples = self.train_samples.sample(n=samples_to_choose)
            self.classifiers[i] = self.get_classifier_tree(self.features_names, ith_train_samples, min_samples)
        for i in range(self.N):
            # todo: calc centroids
            # we take the average per each feature of the N trees
            centroid_vec = [0] * samples_to_choose
            self.centroids[i] =

    # def fit(self, X, Y):
    #   raise NotImplementedError


if __name__ == '__main__':
    # K =
    # N =
    knn = KNNDecisionTree(K=10, N=5)
    knn.fit()
