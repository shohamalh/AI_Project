from KNNForest import *


class ImprovedKNNDecisionTree(KNNDecisionTree):
    # predict = on test group.
    # fit = study
    def __init__(self, N, K, p):
        KNNDecisionTree.__init__(self, N, K, p)  # initializing train and test CSVs.

    def normalize_data(self):
        """
        We normalize using Z-SCORE = (X - μ) / σ
        we calculate the estimator of σ, since it's unbiased (Bessel's correction. Thanks to Introduction to Statistics).
        NOTE: SCIKIT stated that a BIASED estimator is NOT LIKELY to affect the outcome, but I will do so anyway.
        :return: normalized data
        """
        tmp_mean = self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]].mean()
        tmp_sd = self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]].std()
        # normalizing train samples by MinMax
        self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]] = \
            (self.normalized_train_samples[self.normalized_train_samples.columns.values[1:]] - tmp_mean) / tmp_sd
        # normalizing test samples by MinMax, where min and max are from train samples.
        self.test_features_values = (self.test_features_values - tmp_mean) / tmp_sd


if __name__ == '__main__':
    # K =
    # N =
    best_p = 0.3
    highest_acc = 0
    highest_avg_acc = 0
    for i in range(35):
        avg_acc = 0
        p = 0.3 + i * 0.02
        improved_knn = ImprovedKNNDecisionTree(N=10, K=5, p=p)
        for j in range(10):
            improved_knn.fit()
            res_predictions = improved_knn.predict()
            accuracy = improved_knn.accuracy(res_predictions)
            print(j + 1, 'th iteration accuracy:', accuracy, 'probability ', p)
            if accuracy >= highest_acc:
                highest_acc = accuracy
            avg_acc += accuracy
        if avg_acc >= highest_avg_acc:
            highest_avg_acc = avg_acc
            best_p = p
        print('average accuracy of p =', p, ' is', avg_acc)
    print('highest accuracy:', accuracy, ' p:', best_p)
