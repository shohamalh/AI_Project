from ID3 import ID3
from time import time


class CostSensitiveID3(ID3):
    def __init__(self):
        ID3.__init__(self)

    def experiment_loss(self):
        self.experiment('loss')


if __name__ == '__main__':
    id3 = CostSensitiveID3()
    # id3.experiment()
    t = time()
    id3.train()
    print(str(id3.loss()))
    print('took', time() - t)
    t = time()
    id3.train(10)
    print(str(id3.loss()))
    print('took', time() - t)
