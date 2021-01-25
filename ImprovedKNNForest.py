from KNNForest import *
import pandas as pd
from ID3 import Tree


class ImprovedKNNDecisionTree(KNNDecisionTree):
    # predict = on test group.
    # fit = study
    def __init__(self, N=65, K=7, P=0.6555555555555554):
        KNNDecisionTree.__init__(self, N, K, P)  # initializing train and test CSVs.

    @staticmethod
    def _max_feature(features, values: pd.DataFrame):
        b_samples, m_samples = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        m_diagnosis_numbered = (values['diagnosis'] == "M").to_numpy() * 1.0
        b_diagnosis_numbered = 1 - m_diagnosis_numbered
        entropy = ID3._get_entropy(b_samples.shape[0], m_samples.shape[0])
        best_feature = None
        best_info_gain = float('-inf')
        best_feature_threshold = 0
        side_prob = (1 + np.arange(len(m_diagnosis_numbered))) / len(m_diagnosis_numbered)
        for feature in features:
            if feature != 'diagnosis':
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

    def _get_classifier_tree_wrapper(self, samples, min_samples=1, m=12):
        features = self._best_features(samples)[:m]  # best 10 features we will use now.
        best_feature, threshold = self._max_feature(features, samples)
        features.append('diagnosis')
        left = samples.loc[samples[best_feature] < threshold][features]
        right = samples.loc[samples[best_feature] >= threshold][features]  # if == threshold, we go right
        # we take only the left and right of the features sent
        children = (self._get_classifier_tree(left, min_samples, features),
                    self._get_classifier_tree(right, min_samples, features))
        return Tree(best_feature, threshold, children)

    def _get_classifier_tree(self, samples, min_samples=1, features=None):
        values = samples.shape[0]
        m_samples = samples.loc[samples['diagnosis'] == 'M']
        number_m_samples = m_samples.shape[0]
        default = 'B' if (values - number_m_samples) >= (values / 2) else 'M'
        n_default_samples = values - number_m_samples if default == "B" else number_m_samples

        if n_default_samples == values \
                or len(features) == 0 \
                or number_m_samples < min_samples \
                or len(samples) == 0:
            return Tree(None, None, None, default)

        best_feature, threshold = self._max_feature(features, samples)
        left = samples.loc[samples[best_feature] < threshold]
        right = samples.loc[samples[best_feature] >= threshold]  # if == threshold, we go right
        if len(left) == 0 or len(right) == 0:
            return Tree(None, None, None, default)

        children = (self._get_classifier_tree(left, min_samples, features),
                    self._get_classifier_tree(right, min_samples, features))
        return Tree(best_feature, threshold, children)

    def _best_features(self, values: pd.DataFrame):
        """
        :param features:
        :param values:
        :return: a list of sorted features and their IGs, by IG, descending.
        """
        features = self.features_names
        b_samples, m_samples = values.loc[values['diagnosis'] == 'B'], values.loc[values['diagnosis'] == 'M']
        m_diagnosis_numbered = (values['diagnosis'] == "M").to_numpy() * 1.0
        b_diagnosis_numbered = 1 - m_diagnosis_numbered
        entropy = ID3._get_entropy(b_samples.shape[0], m_samples.shape[0])
        side_prob = (1 + np.arange(len(m_diagnosis_numbered))) / len(m_diagnosis_numbered)
        features_ig = []  # list of (IG, feature)
        for feature in features:
            info_gain, f_values, f_idxs = ID3._calculate_feature_ig(feature, values, m_diagnosis_numbered,
                                                                    b_diagnosis_numbered, side_prob, entropy)
            features_ig.append((feature, info_gain.argmax()))
        features_ig = sorted(features_ig, key=lambda x: x[1], reverse=True)  # the best 10 features
        ret = [f[0] for f in features_ig]
        return ret

    def fit(self, train_samples=None, size_of_train_group=0, m=12):
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
            self.classifiers[tree] = self._get_classifier_tree_wrapper(ith_train_samples, m=m)

    def _predict_single2(self, sample_to_classify, sorted_centroid_idxs):
        """
        This method classifies a single sample according to the K-nearest centroids.
        :param sample_to_classify: the samples to classify
        :return: a prediction of 'M' or 'B'.
        """
        distances = [(spatial.distance.euclidean(sample_to_classify, centroid), idx)  # (distance, index)
                     for idx, centroid in enumerate(self.centroids[0])]
        # distances is an array of the euclidean distances from the centroids
        sorted_distances = sorted(distances)[:self.K]  # list of (distance, index). tuple sort.

        weights = [1 / (dist[0] ** 2) for dist in sorted_distances]  # weights only vector
        weights = [weight / sum(weights) for weight in weights]  # normalized weights vector
        k_trees_diagnosis = []
        for (_, ind), weight in zip(sorted_distances, weights):
            classification = KNNDecisionTree._classify(sample_to_classify, self.classifiers[ind])  # 'M', 'B'
            classification = (classification == 'M') * 1.0
            k_trees_diagnosis.append(classification * weight)
        prediction = np.sum(k_trees_diagnosis) > 0.5
        return 'M' if prediction else 'B'


if __name__ == '__main__':
    """
    best_p = 0.3
    highest_acc = 0
    highest_avg_acc = 0
    for i in range(9):
        avg_acc = 0
        p = 0.3 + i * 0.05
        knn_tree = KNNDecisionTree(N=10, K=8, P=p)
        for j in range(10):
            knn_tree.fit()
            res_predictions = knn_tree.predict()
            accuracy = knn_tree.accuracy(res_predictions)
            print(j + 1, 'th iteration accuracy:', accuracy, 'probability ', p)
            if accuracy >= highest_acc:
                highest_acc = accuracy
            avg_acc += accuracy
        avg_acc = avg_acc / 10
        if avg_acc >= highest_avg_acc:
            highest_avg_acc = avg_acc
            best_p = p
        print('average accuracy of p =', p, ' is', avg_acc)
    print('highest average accuracy:', highest_avg_acc, ' p:', best_p)
    """
    improved_knn = ImprovedKNNDecisionTree()
    improved_knn.fit()
    res_predictions = improved_knn.predict()
    accuracy = improved_knn.accuracy(res_predictions)
    print(accuracy)
    # knn_experiment(print_graph=True)
    # best_avg_acc = 0
    # best_m = 1
    # for m in range(1, 31):
    #    avg_acc = 0
    #    for i in range(5):
    #        improved_knn = ImprovedKNNDecisionTree()
    #        improved_knn.fit(m=m)
    #        res_predictions = improved_knn.predict()
    #        accuracy = improved_knn.accuracy(res_predictions)
    #        avg_acc += accuracy
    #        print(accuracy)
    #    avg_acc /= 5
    #    print(f'average acc for m = {m} is {avg_acc}')
    #    if avg_acc >= best_avg_acc:
    #        best_avg_acc = avg_acc
    #        best_m = m
    # print(f'best_m = {best_m}, best_avg_acc = {best_avg_acc}')
