import tensorflow as tf
import numpy as np
import time
import progressbar


class Solver(object):
    """
    A generic solver for any classification problem
    """

    def __init__(self, model, data, dtype, graph, sess, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']  # N x 1
        self.num_training_samples = self.X_train.shape[0]

        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.num_val_samples = self.X_val.shape[0]

        self.sess = sess
        self.graph = graph

        self.num_epochs = kwargs.pop('num_epochs', 10)  # default is 10 epochs
        self.batch_size = kwargs.pop('batch_size', 1)  # default is 1 sample per mini-batch
        self.optimizer = kwargs.pop('optimizer', 'sgd')  # default is sgd
        self.optimizer_config = kwargs.pop('optimizer_config', {'learning_rate': 0.01, 'decay_rate': 1.0})
        self.learning_rate = tf.Variable(initial_value=self.optimizer_config['learning_rate'], trainable=False)
        if 'decay_rate' in self.optimizer_config:
            self.decay_rate = tf.constant(self.optimizer_config['decay_rate'], dtype=dtype)
        else:
            self.decay_rate = tf.constant(1.0, dtype=dtype)
        self.export_summary = kwargs.pop('export_summary', False)
        self.summary_config = kwargs.pop('summary_config', None)

    def build_computation_graph(self):
        print 'build computation graph...'
        with self.graph.as_default():
            tf.set_random_seed(1337)  # for reproduction
            with tf.name_scope('steps'):
                global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')
                increment_step = global_step.assign_add(1)
            with tf.name_scope('inputs'):
                X, Y = self.model.X, self.model.Y
            with tf.name_scope('loss'):
                loss = self.model.loss(X, Y)
                check_accuracy = self.model.check_accuracy(X, Y)
            with tf.name_scope('optimizer'):
                optimizer = None
                if self.optimizer == 'adam':
                    beta1 = self.optimizer_config.get('beta1', 0.9)
                    beta2 = self.optimizer_config.get('beta2', 0.999)
                    epsilon = self.optimizer_config.get('epsilon', 1e-8)
                    optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2, epsilon)
                elif self.optimizer == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                elif self.optimizer == 'momentum':
                    momentum = self.optimizer_config.get('momentum', 0.9)
                    use_nesterov = self.optimizer_config.get('use_nesterov', True)
                    optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum, use_nesterov=use_nesterov)
                elif self.optimizer == 'rmsprop':
                    decay = self.optimizer_config.get('decay', 0.9)
                    momentum = self.optimizer_config.get('momentum', 0.0)
                    epsilon = self.optimizer_config.get('epsilon', 1e-10)
                    centered = self.optimizer_config.get('centered', False)
                    optimizer == tf.train.RMSPropOptimizer(self.learning_rate, decay=decay, momentum=momentum,
                                                           epsilon=epsilon, centered=centered)
                else:
                    raise ValueError('Unknown optimizer')
                objective = optimizer.minimize(loss)
            with tf.name_scope('summaries'):
                tf.summary.scalar('loss', loss)
            with tf.name_scope('global_ops'):
                # only call variable initialization after finishing building the graph
                init = tf.global_variables_initializer()
                merged_summaries = tf.summary.merge_all()

            return init, increment_step, merged_summaries, loss, objective, check_accuracy

    def train(self):
        """
        train the model using mini-batch, the update rule is gradient descent
        """
        init, increment_step, merged_summaries, loss, optimizer, check_accuracy = self.build_computation_graph()
        self.sess.run(init)

        # only call file writer after constructing the graph
        writer = None
        if self.export_summary:
            if 'path' not in self.summary_config:
                raise KeyError('Must specify the path to export summary')
            path = self.summary_config['path']
            writer = tf.summary.FileWriter(path, self.graph)

        # main training loop
        for epoch in xrange(self.num_epochs):

            start = time.time()

            arr = np.arange(self.num_training_samples)
            np.random.shuffle(arr)
            batch_mask_list = np.array_split(arr, self.num_training_samples / self.batch_size)
            total_loss = 0.0
            step = self.sess.run(increment_step)

            print 'Epoch: %d/%d' % (step, self.num_epochs)

            bar = progressbar.ProgressBar(redirect_stdout=False, max_value=self.num_training_samples)
            current_processed_num = 0

            for batch_mask in batch_mask_list:
                X = self.X_train[batch_mask]
                Y = self.y_train[batch_mask]
                feed_dict = {self.model.X: X, self.model.Y: Y}
                if writer is not None:
                    _, l, summary = self.sess.run([optimizer, loss, merged_summaries], feed_dict=feed_dict)
                    writer.add_summary(summary, global_step=step)
                else:
                    _, l = self.sess.run([optimizer, loss], feed_dict=feed_dict)

                total_loss += l

                current_processed_num += len(batch_mask)
                bar.update(current_processed_num)

            bar.finish()
            # decay the learning rate
            self.learning_rate.assign(tf.multiply(self.learning_rate, self.decay_rate))

            # training date accuracy
            correct_predicted = 0
            for batch_mask in batch_mask_list:
                X = self.X_train[batch_mask]
                Y = self.y_train[batch_mask]
                feed_dict = {self.model.X: X, self.model.Y: Y}
                correct_predicted_batch = self.sess.run(check_accuracy, feed_dict=feed_dict) * len(batch_mask)

                correct_predicted += int(correct_predicted_batch)

            avg_loss = total_loss
            training_accuracy = float(correct_predicted) / self.num_training_samples

            # validation data
            batch_mask_list = np.array_split(np.arange(self.num_val_samples), self.num_val_samples / self.batch_size)
            val_loss, val_correct_predicted = 0.0, 0
            for batch_mask in batch_mask_list:
                X = self.X_val[batch_mask]
                Y = self.y_val[batch_mask]
                feed_dict = {self.model.X: X, self.model.Y: Y}
                val_loss_batch, val_accuracy = self.sess.run([loss, check_accuracy], feed_dict=feed_dict)
                val_loss += val_loss_batch
                val_correct_predicted += val_accuracy * len(batch_mask)

            avg_val_loss = val_loss
            val_accuracy = float(val_correct_predicted) / self.num_val_samples

            elapse = time.time() - start

            print 'loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f - total elapse: %.4fs' % \
                  (avg_loss, training_accuracy, avg_val_loss, val_accuracy, elapse)

        if writer is not None:
            writer.flush()
            writer.close()

