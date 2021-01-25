from ID3 import *


class CostSensitiveID3(ID3):
    def __init__(self):
        ID3.__init__(self)

    def _get_classifier_tree(self, samples, min_samples=1, features=None):
        if features is None:
            features = self.features_names
        values = samples.shape[0]
        m_samples = (samples.loc[samples['diagnosis'] == 'M']).shape[0]
        b_samples = (samples.loc[samples['diagnosis'] == 'B']).shape[0]
        default = 'B' if (samples.loc[samples['diagnosis'] == 'B']).shape[0] >= samples.shape[0] / 2 else 'M'
        if (samples.loc[samples['diagnosis'] == default]).shape[0] == values \
                or features.shape[0] == 0 or (m_samples < min_samples and b_samples - m_samples <= 2):
            node = Tree(None, None, None)
            if m_samples < min_samples and b_samples - m_samples <= 2:
                node.classification = 'M'
            else:
                node.classification = default
            return node

        best_feature, threshold = self._max_feature(features, samples)
        left = samples.loc[samples[best_feature] < threshold]
        right = samples.loc[samples[best_feature] >= threshold]
        children = (self._get_classifier_tree(left, min_samples),
                    self._get_classifier_tree(right, min_samples))
        return Tree(best_feature, threshold, children)


if __name__ == '__main__':
    cost_sensitive_id3 = CostSensitiveID3()
    # cost_sensitive_id3.experiment(predict_or_loss='loss', print_graph=True)
    # exit(0)
    cost_sensitive_id3.fit(20)
    res_predictions = cost_sensitive_id3.predict()
    loss = cost_sensitive_id3.loss(res_predictions)
    print(loss)
