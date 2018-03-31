import os
from glob import glob

import numpy as np
import cv2


class DataManager(object):
    def __init__(self, class_n, batch_size, data_width, data_height, data_channel):
        self.num_class = class_n
        self.batch_size = batch_size

        self.data_width = data_width  # 이미지의 사이즈가 28, 28 이고, Gray Scale 이기 때문에 Channel은 1로 주었다.
        self.data_height = data_height
        self.data_channel = data_channel

        train_path_list = glob('./mnist_png/training/*/*.png')
        test_path_list = glob('./mnist_png/testing/*/*.png')

        # Data Path에 있는 각 Label 이름들을 가져와 Label로 만들어줌. 예) 0, 1, 2, 3, 4, 5 ... 9
        train_label_list = [int(os.path.dirname(path).split('\\')[-1]) for path in train_path_list]
        test_label_list = [int(os.path.dirname(path).split('\\')[-1]) for path in test_path_list]

        # zip으로 이미지 경로와 label 번호를 묶고, list로 만들어주었다.
        self.train_list = list(zip(train_path_list, train_label_list))
        self.test_list = list(zip(test_path_list, test_label_list))

    def _encode_onehot(self, label, n_class=10):
        # 여기서 label은 1이나 2, 3과 같이 Encoding이 되어있지 않은 Label을 받게 된다.
        # Return은 1은 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        label_onehot = np.zeros(n_class)
        label_onehot[label] = 1
        return label_onehot

    def _read_image(self, data_path):
        # 파라미터를 0으로 준 것은 MNIST가 GrayScale 이기 때문이다.
        image = cv2.imread(data_path, 0)
        image = image.reshape((self.data_width, self.data_height, self.data_channel))
        return image

    # 아래의 함수로 매 Batch 학습 때 마다 data를 입력 해줄 것 이다.
    def input_data(self, batch_n, mode='train'):
        label_batch = np.zeros((self.batch_size, self.num_class))
        image_batch = np.zeros((self.batch_size, self.data_width, self.data_height, self.data_channel))

        if mode == 'train':
            batch_list = self.train_list[batch_n * self.batch_size:(batch_n + 1) * self.batch_size]
        else:
            batch_list = self.test_list[batch_n * self.batch_size:(batch_n + 1) * self.batch_size]

        for i, (data_path, label_n) in enumerate(batch_list):
            label_batch[i] = self._encode_onehot(label_n)
            image_batch[i] = self._read_image(data_path)

        return image_batch, label_batch
