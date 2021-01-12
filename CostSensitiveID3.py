from ID3 import *


class CostSensitiveID3(ID3):
    def __init__(self):
        ID3.__init__(self)

    @staticmethod
    def get_classifier_tree(features, values, min_samples=1):
        samples = values.shape[0]
        m_samples = (values.loc[values['diagnosis'] == 'M']).shape[0]
        b_samples = (values.loc[values['diagnosis'] == 'B']).shape[0]
        default = 'B' if (values.loc[values['diagnosis'] == 'B']).shape[0] >= values.shape[0] / 2 else 'M'
        if (values.loc[values['diagnosis'] == default]).shape[0] == samples or features.shape[0] == 0:
            node = Node(None, None, None)
            node.set_classification(default)
            return node

        if m_samples < min_samples and b_samples - m_samples <= 5:
            node = Node(None, None, None)
            node.set_classification('M')
            return node

        best_feature, threshold = CostSensitiveID3.max_feature(features, values)
        left = values.loc[values[best_feature] < threshold]
        right = values.loc[values[best_feature] >= threshold]
        children = (CostSensitiveID3.get_classifier_tree(features, left, min_samples),
                    CostSensitiveID3.get_classifier_tree(features, right, min_samples))
        return Node(best_feature, threshold, children)


if __name__ == '__main__':
    id3 = CostSensitiveID3()
    # id3.experiment()
    id3.train(10)
    print(str(id3.loss()))
    # print(str(id3.test()))
