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
        spatial.distance.cdist

        distances = [(spatial.distance.euclidean(sample_to_classify, centroid), idx)  # (distance, index)
                     for idx, centroid in enumerate(self.centroids[0])]
        # distances is an array of the euclidean distances from the centroids
        k_nearest = sorted(distances)[:self.K]  # list of (distance, index). tuple sort.
        k_trees_diagnosis = []
        for _, ind in k_nearest:
            k_trees_diagnosis.append(KNNDecisionTree._classify(sample_to_classify, self.classifiers[ind]))
        return max(k_trees_diagnosis, key=k_trees_diagnosis.count)  # we classify according to the Majority Class.

        # prediction = max(k_trees_diagnosis, key=k_trees_diagnosis.count)  # we classify according to what we have more.
        # prediction = np.sum(k_trees_diagnosis) > 0.5
        # return 'M' if prediction else 'B'
        # weights = [1 / dist[0] for dist in k_nearest]  # weights only vector
        # weights = [weight / sum(weights) for weight in weights]  # normalized weights vector
        # k_trees_diagnosis = []
        # for (_, ind), weight in zip(k_nearest, weights):
        #     classification = KNNDecisionTree._classify(sample_to_classify, self.classifiers[ind])  # 'M', 'B'
        #     classification = (classification == 'M') * 1.0
        #     k_trees_diagnosis.append(classification * weight)
        # prediction = np.sum(k_trees_diagnosis) > 0.5
        # return 'M' if prediction else 'B'


def knn_experiment(print_graph=False):
    best_avg_acc = 0
    best_N = 0
    best_K = 0
    best_p = 0
    p_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    ks = [3, 5, 7, 11]
    ns = np.linspace(max(ks), max(ks) * 10, 10, dtype=int)
    for K in ks:  # we check every even N from 10 to 30, just so it won't run for ages.
        for N in ns:
            test_acc = []
            for p in p_vals:  # we also try to check the p parameter
                avg_acc = 0
                cum_acc = 0
                experiment_knn = ImprovedKNNDecisionTree(N=N, K=K, P=p)
                kf = KFold(n_splits=5, shuffle=True, random_state=208501684)
                for fold_id, (train_index, test_index) in enumerate(kf.split(experiment_knn.train_samples)):
                    acc = 0
                    train_samples = experiment_knn.train_samples.loc[train_index]
                    test_samples = experiment_knn.train_samples.loc[test_index]
                    experiment_knn.fit(train_samples, train_samples.shape[0] - 1)  # creating trees
                    predictions = experiment_knn.predict(test_samples)
                    acc = experiment_knn.accuracy(predictions, test_samples)
                    # print(f'N = {N}, K = {K}, p = {p}, Fold = {fold_id} Accuracy = {acc}')
                    cum_acc += acc
                avg_acc = cum_acc / 5
                print(f'N = {N}, K = {K}, p = {p}, acg_acc = {avg_acc}')
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
    knn_experiment(True)
    exit(0)
    improved_knn = ImprovedKNNDecisionTree(N=10, K=7)
    avg_acc = 0
    for i in range(2):
        improved_knn.fit()
    res_predictions = improved_knn.predict()
    accuracy = improved_knn.accuracy(res_predictions)
    print(i + 1, 'th iteration accuracy:', accuracy)
    avg_acc += accuracy
    print('avg acc', avg_acc / 10)
