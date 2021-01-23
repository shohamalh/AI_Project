from KNNForest import *


class ImprovedKNNDecisionTree(KNNDecisionTree):
    # predict = on test group.
    # fit = study
    def __init__(self, N, K, P=0.35):
        KNNDecisionTree.__init__(self, N, K, P)  # initializing train and test CSVs.

    def _predict_single(self, sample_to_classify):
        """
        This method classifies a single sample according to the K-nearest centroids.
        :param sample_to_classify: the samples to classify
        :return: a prediction of 'M' or 'B'.
        """
        distances = [(spatial.distance.correlation(sample_to_classify, centroid), idx)  # (distance, index) # MANHATTAN
                     for idx, centroid in enumerate(self.centroids[0])]
        # distances is an array of the euclidean distances from the centroids
        k_nearest = sorted(distances)[:self.K]
        k_trees_diagnosis = []
        for _, ind in k_nearest:
            k_trees_diagnosis.append(KNNDecisionTree._classify(sample_to_classify, self.classifiers[ind]))
        prediction = max(k_trees_diagnosis, key=k_trees_diagnosis.count)  # we classify according to what we have more.
        return prediction


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
    improved_knn = ImprovedKNNDecisionTree(N=10, K=7)
    avg_acc = 0
    for i in range(10):
        improved_knn.fit()
        res_predictions = improved_knn.predict()
        accuracy = improved_knn.accuracy(res_predictions)
        print(i + 1, 'th iteration accuracy:', accuracy)
        avg_acc += accuracy
    print('avg acc', avg_acc / 10)
