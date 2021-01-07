import pandas as pd
import numpy as np
from KNN import KNN
from DT_epsilon import ID3_epsilon


class KNN_epsilon(ID3_epsilon):
    def __init__(self, train, test):
        ID3_epsilon.__init__(self, train, test)
        predictors_train, targets = KNN.get_data(train)
        self.train = (predictors_train - predictors_train.min()) / (predictors_train.max() - predictors_train.min())

        self.train.insert(loc=0, column='diagnosis', value=targets.values)

        predictors_test, targets = KNN.get_data(test)
        self.test = (predictors_test - predictors_train.min()) / (predictors_train.max() - predictors_train.min())

        self.test.insert(loc=0, column='diagnosis', value=targets.values)

    def fit_epsilon_KNN(self):
        v = np.std(self.train.values, axis=0, ddof=0)
        epsilon = v * 0.1
        keys = list(self.train.keys())
        final_epsilon = pd.DataFrame()
        for i in range(len(keys)):
            final_epsilon[keys[i]] = [epsilon[i]]
        final_epsilon = final_epsilon.apply(pd.to_numeric, errors='coerce')
        correct = 0
        for i in range(self.test.shape[0]):
            self.examples_at_leafs = pd.DataFrame(columns=list(self.train.keys()))
            self.DT_Classify_epsilon(self.test.iloc[[i]], self.tree, final_epsilon)
            if self.examples_at_leafs.shape[0] < 9:
                classification = 1 if (self.examples_at_leafs.loc[self.examples_at_leafs['diagnosis'] == 1]).shape[
                                          0] >= 0.5 * self.examples_at_leafs.shape[0] else 0
            else:
                knn = KNN(self.examples_at_leafs, self.test.iloc[[i]], fromFile=False)
                classification = knn.predict(9)[0][0]

            if classification == self.test.iloc[[i]]['diagnosis'].values[0]:
                correct += 1
        return correct / self.test.shape[0]


if __name__ == '__main__':
    instance = KNN_epsilon('train', 'test')
    instance.buildTree(9)
    print(str(instance.fit_epsilon_KNN() * 100))
