from ID3 import *


class CostSensitiveID3(ID3):
    def __init__(self):
        ID3.__init__(self)

    @staticmethod
    def get_classifier_tree(features, samples, min_samples=1):
        values = samples.shape[0]
        m_samples = (samples.loc[samples['diagnosis'] == 'M']).shape[0]
        b_samples = (samples.loc[samples['diagnosis'] == 'B']).shape[0]
        default = 'B' if (samples.loc[samples['diagnosis'] == 'B']).shape[0] > samples.shape[0] / 2 else 'M'
        if (samples.loc[samples['diagnosis'] == default]).shape[0] == values or features.shape[0] == 0:
            node = Tree(None, None, None)
            node.set_classification(default)
            return node

        if m_samples <= min_samples and b_samples - m_samples <= 5:
            node = Tree(None, None, None)
            node.set_classification('M')
            return node

        best_feature, threshold = CostSensitiveID3.max_feature(features, samples)
        left = samples.loc[samples[best_feature] <= threshold]
        right = samples.loc[samples[best_feature] > threshold]
        children = (CostSensitiveID3.get_classifier_tree(features, left, min_samples),
                    CostSensitiveID3.get_classifier_tree(features, right, min_samples))
        return Tree(best_feature, threshold, children)


if __name__ == '__main__':
    id3 = CostSensitiveID3()
    # id3.experiment()
    id3.fit(10)
    print(str(id3.loss()))
    # print(str(id3.predict()))
