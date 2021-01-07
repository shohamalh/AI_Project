from DT import ID3
from KNN import KNN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # section 3
    instance1, instance2, instance3 = ID3('train', 'test'), ID3('train', 'test'), ID3('train', 'test')
    instance1.buildTree(3), instance2.buildTree(9), instance3.buildTree(27)
    DT3, DT9, DT27 = instance1.fit(), instance2.fit(), instance3.fit()
    DT3_TE, DT9_TE, DT27_TE = instance1.fit(training_error=True), instance2.fit(training_error=True), instance3.fit(training_error=True)
    x_points = [3, 9, 27]
    y_points = [DT3, DT9, DT27]
    plt.title('Accuracy over minimum x to split')
    plt.xlabel('x')
    plt.ylabel('Accuracy')
    plt.plot(x_points, y_points, marker='o')
    for txt, x, y in zip([round(DT3, 3), round(DT9, 3), round(DT27, 3)], x_points, y_points):
        plt.annotate(txt, (x, y))
    plt.show()
    plt.close()

    # section 9
    instance = KNN('train', 'test')
    test_predictors = KNN.get_data('test')[0]
    x_points = [1, 3, 9, 27]
    y_points = []
    plt.title('Accuracy over k')
    plt.xlabel('x')
    plt.ylabel('Accuracy')

    for k in [1, 3, 9, 27]:
        predictions = instance.predict(k)
        correct = list(filter(lambda z: z[1] == 1, predictions))
        y_points.append(len(correct) / test_predictors.shape[0])

    plt.plot(x_points, y_points, marker='o')
    for txt, x, y in zip([round(point, 3) for point in y_points], x_points, y_points):
        plt.annotate(txt, (x, y))
    plt.show()
    plt.close()
