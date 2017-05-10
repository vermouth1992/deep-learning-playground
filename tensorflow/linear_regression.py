"""
A linear regression class
"""

from model import MachineLearningModel
import tensorflow as tf


class LinearRegression(MachineLearningModel):
    def __init__(self, dtype, input_shape):
        MachineLearningModel.__init__(self, dtype=dtype)
        self.X = tf.placeholder(dtype=self.dtype, shape=[None, input_shape], name='X')
        self.Y = tf.placeholder(dtype=self.dtype, shape=[None], name='Y')
        self.W = tf.Variable(tf.zeros(input_shape), dtype=self.dtype, name='weights')  # Dx1 matrix
        self.b = tf.Variable(0.0, dtype=self.dtype, name='bias')

    def inference(self, X):
        return tf.multiply(X, self.W) + self.b

    def loss(self, X, Y):
        """
        :return: scores (test) or loss (train)
        """
        Y_predicted = self.inference(X)
        return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

    def evaluate(self, sess, X):
        return sess.run([self.inference(X)])


# example
if __name__ == '__main__':
    """
    To build a simple linear regression model
    """
    import numpy as np
    import xlrd
    import matplotlib.pyplot as plt

    from solver import Solver

    DATA_FILE = '../data/fire_theft.xls'

    # Step 1: read in data from the .xls file
    book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1

    X_train = data[:, 0:1]
    y_train = data[:, 1:2]

    training_data = {'X_train': X_train, 'y_train': y_train}

    graph = tf.Graph()

    with tf.Session(graph=graph) as sess:
        linear_regression_model = LinearRegression(dtype=tf.float32, input_shape=[1])
        solver = Solver(linear_regression_model, training_data, dtype=np.float32, graph=graph, sess=sess,
                        optimizer='momentum', optimizer_config={'learning_rate': 1e-4},
                        batch_size=n_samples, num_epochs=150,
                        export_summary=False, summary_config={'path': './my_graph/linear_regression'})
        solver.train()

        w_value, b_value = sess.run([linear_regression_model.W, linear_regression_model.b])

        # analytical solution
        X_train_add_bias = np.hstack((X_train, np.ones((n_samples, 1))))
        theta = np.linalg.inv(X_train_add_bias.T.dot(X_train_add_bias)).dot(X_train_add_bias.T).dot(y_train)
        theoretical_w_value = theta[0:-1][0]
        theoretical_b_value = theta[-1][0]
        # compare the two
        print 'theoretical w', theoretical_w_value
        print 'gradient w', w_value
        print 'theoretical b', theoretical_b_value
        print 'gradient b', b_value

        is_plot = False

        if is_plot:
            X, Y = data.T[0], data.T[1]
            plt.plot(X, Y, 'bo', label='Real data')
            plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
            plt.legend()
            plt.show()
