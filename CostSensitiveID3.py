from DT import ID3
import pandas as pd
import numpy as np


class ID3_epsilon(ID3):
    def __init__(self, train, test):
        ID3.__init__(self, train, test)
        self.examples_at_leafs = pd.DataFrame(columns=list(self.train.keys()))

    def DT_Classify_epsilon(self, o, Tree, epsilon):
        if not Tree[1]:  # no children
            self.examples_at_leafs = pd.concat([self.examples_at_leafs, Tree[3]], ignore_index=True)
        elif abs(o[Tree[0]].values[0] - Tree[1][0][0]) <= epsilon[Tree[0]].values[0]:  # |xi-vi|<=ei
            self.DT_Classify_epsilon(o, Tree[1][0][1], epsilon)
            self.DT_Classify_epsilon(o, Tree[1][1][1], epsilon)
        elif o[Tree[0]].values[0] <= Tree[1][0][0]:  # left son
            self.DT_Classify_epsilon(o, Tree[1][0][1], epsilon)
        else:
            self.DT_Classify_epsilon(o, Tree[1][1][1], epsilon)  # right son

    def fit_epsilon(self):
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
            classification = 1 if (self.examples_at_leafs.loc[self.examples_at_leafs['diagnosis'] == 1]).shape[
                                      0] >= 0.5 * self.examples_at_leafs.shape[0] else 0
            if classification == self.test.iloc[[i]]['diagnosis'].values[0]:
                correct += 1
        return correct / self.test.shape[0]


if __name__ == '__main__':
    instance = ID3_epsilon('train', 'test')
    instance.buildTree(9)
    print(str(instance.fit_epsilon() * 100))
