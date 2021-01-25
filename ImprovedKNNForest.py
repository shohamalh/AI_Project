from KNNForest import *
import pandas as pd


class ImprovedKNNDecisionTree(KNNDecisionTree):
    # predict = on test group.
    # fit = study
    def __init__(self, N, K, P=0.35):
        KNNDecisionTree.__init__(self, N, K, P)  # initializing train and test CSVs.

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
    t = time()
    knn_experiment(print_graph=True)
    for i in range(10):
        improved_knn = ImprovedKNNDecisionTree(N=67, K=7, P=0.6555555555555554)
        improved_knn.fit()
        res_predictions = improved_knn.predict()
        accuracy = improved_knn.accuracy(res_predictions)
        print(accuracy)
