import heapq
import pandas as pd
from scipy import spatial


class KNN:
    def __init__(self, train, test, fromFile=True):
        if fromFile:
            predictors, targets = KNN.get_data(train)
            self.predictors_data, self.predictors_targets = KNN.get_data(test)
            self._train_min = predictors.min().to_numpy()
            self._train_max = predictors.max().to_numpy()
            self._final_train = (predictors - predictors.min()) / (predictors.max() - predictors.min())
            self._final_train = self._final_train.values
        else:
            predictors = train[train.columns.values[1:]]
            targets = train[train.columns.values[0]]
            predictors = predictors.apply(pd.to_numeric, errors='coerce')
            targets = targets.apply(pd.to_numeric, errors='coerce')
            self.predictors_data = test[test.columns.values[1:]]
            self.predictors_targets = test[test.columns.values[0]]
            self.predictors_data = self.predictors_data.apply(pd.to_numeric, errors='coerce')
            self.predictors_targets = self.predictors_targets.apply(pd.to_numeric, errors='coerce')
            self._final_train = predictors.values
            self._train_min = None  # do not use
            self._train_max = None  # do not use

        self._train_targets = targets.values

    def predict(self, k):
        to_return = []
        for predictors, index in zip(self.predictors_data.values, range(self.predictors_data.shape[0])):
            prediction = self._predict_single(predictors, k)
            good_one = 1 if prediction == self.predictors_targets.iloc[[index]].values[0] else 0
            to_return.append((prediction, good_one))
        return to_return

    def _predict_single(self, features, k):
        if self._train_min is not None:
            features = (features - self._train_min) / (self._train_max - self._train_min)
        distances = [(spatial.distance.euclidean(features, self_train), self_train_target)
                     for self_train, self_train_target in zip(self._final_train, self._train_targets)]
        nearest = heapq.nsmallest(k, distances, key=lambda x: x[0])
        nearest_targets = [e[1] for e in nearest]
        target = max(set(nearest_targets), key=nearest_targets.count)
        return target

    @staticmethod
    def get_data(csv):
        data = pd.read_csv(f'{csv}.csv')
        features = data[data.columns.values[1:]]
        results = data[data.columns.values[0]]
        features = features.apply(pd.to_numeric, errors='coerce')
        results = results.apply(pd.to_numeric, errors='coerce')

        return features, results


if __name__ == '__main__':
    knn = KNN('train', 'test')
    test_predictors = KNN.get_data('test')[0]
    predictions = knn.predict(9)
    correct = list(filter(lambda x: x[1] == 1, predictions))
    print(str(len(correct) / test_predictors.shape[0] * 100))
