from datetime import datetime

from mnist_basic.data_manager import *
from mnist_basic.model import *
from mnist_basic.utill import *


class TrainModel(DataManager, BuildModel):
    def __init__(self, mode='train', display_step=1, learning_rate=0.001, train_epochs=5, batch_size=128, class_n=10, data_width=28, data_height=28, data_channel=1):
        DataManager.__init__(self, class_n, batch_size, data_width, data_height, data_channel)
        self.display_step = display_step

        self.learning_rate = learning_rate
        self.train_epochs = train_epochs

        self.num_train_data = len(self.train_list)
        self.num_test_data = len(self.test_list)

        self.train_batches_per_epoch = int(self.num_train_data // self.batch_size)
        self.test_batches_per_epoch = int(self.num_test_data // self.batch_size)

        self.glob_step = 0

        self.mode = mode
        if self.mode == 'train':
            self.is_training = True
            self.keep_prob = 0.7
        elif self.mode == 'test':
            self.is_training = False
            self.keep_prob = 1.
        else:
            raise ValueError("Please Choose between 'train' / 'test'")

    def train(self):
        g = tf.Graph()
        with g.as_default():
            is_training = tf.placeholder_with_default(tf.constant(True), None, name='is_training')
            keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')

            image = tf.placeholder(tf.float32, [None, self.data_width, self.data_height, self.data_channel], 'input')
            label = tf.placeholder(tf.int8, [None, self.num_class], 'label')

            model = BuildModel(is_training, keep_prob, self.num_class)
            logit = model.inference(image)

            OUTPUT = tf.nn.softmax(logit, name="output")

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))
            optm = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
            corr = tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1))  # 예측한 결과( y' )가 정답인 y하고 맞는지 확인한다.
            accr = tf.reduce_mean(tf.cast(corr, tf.float32))  # 맞는 갯수의 평균으로 정확도를 계산한다.
            init = tf.global_variables_initializer()  # 모든 Variables들을 초기화하는 것을 만든다.

            with tf.Session() as sess:
                # 학습을 시작하기 전에 Variable 들을 초기화 한다.
                sess.run(init)
                print("Start Training!", datetime.now())
                eval_step = 0
                for epoch in range(self.train_epochs):
                    avg_cost = 0.
                    avg_accr = 0.

                    np.random.shuffle(self.train_list)
                    for batch_n in range(self.train_batches_per_epoch):
                        image_batch, label_batch = self.input_data(batch_n)

                        train_feed_dict = {image: image_batch, label: label_batch, keep_prob: self.keep_prob}
                        sess.run(optm, feed_dict=train_feed_dict)

                        # Cost 평균값 업데이트
                        cost_value, accr_value = sess.run([cost, accr], feed_dict=train_feed_dict)
                        avg_cost += cost_value / self.train_batches_per_epoch
                        avg_accr += accr_value / self.train_batches_per_epoch

                        progress_bar(self.train_batches_per_epoch, batch_n, state_msg='Average Cost : %.4f, Average Accuracy : %.4f' % (avg_cost, avg_accr))

                        self.glob_step += 1

                    # Display logs per epoch step
                    if epoch % self.display_step == 0:
                        eval_batch_n = (len(self.train_list) * eval_step) % len(self.test_list)
                        test_image_batch, test_label_batch = self.input_data(eval_batch_n, mode='test')
                        test_feed_dict = {image: test_image_batch, label: test_label_batch, keep_prob: self.keep_prob, is_training: False}
                        print("Epoch: %03d/%03d, Cost: %.9f" % (epoch + 1, self.train_epochs, avg_cost))
                        train_acc = sess.run(accr, feed_dict=train_feed_dict)
                        print(" Training Accuracy: %.3f" % (train_acc))
                        test_acc = sess.run(accr, feed_dict=test_feed_dict)
                        print(" Test Accuracy: %.3f" % (test_acc))
                        eval_step += 1

                    save_ckpt(sess, self.glob_step)

                graph_def = g.as_graph_def()
                tf.train.write_graph(graph_def, './', 'mnist_model_graph.pb', as_text=False)

                print("Done training at : ", datetime.now(), "\n")


def main(argv=None):
    optimizer = TrainModel(FLAGS.mode, FLAGS.display_step, FLAGS.learning_rate, FLAGS.train_epochs, FLAGS.batch_size)
    optimizer.train()


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('mode', 'train',
                               """Choose between 'train'/'test'""")
    tf.app.flags.DEFINE_integer('display_step', 10,
                               """Display step per Epochs""")
    tf.app.flags.DEFINE_float('learning_rate', 0.001,
                               """Learning Rate""")
    tf.app.flags.DEFINE_float('train_epochs', 1,
                               """Number of Train Epochs""")
    tf.app.flags.DEFINE_integer('batch_size', 128,
                               """Train Batch Size""")

    tf.app.run()
