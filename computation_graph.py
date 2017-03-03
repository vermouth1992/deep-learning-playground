"""
This file demonstrates a basic computation graph running in Tensorflow
"""

import tensorflow as tf
import numpy as np

graph = tf.Graph()

# create a graph
with graph.as_default():

    with tf.name_scope('variables'):
        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')
        total_output = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False, name='total_output')

    with tf.name_scope('transformation'):

        with tf.name_scope('input'):
            a = tf.placeholder(tf.float32, shape=[None], name='input_placeholder_a')

        with tf.name_scope('intermediate_layer'):
            b = tf.reduce_prod(a, name='product_b')
            c = tf.reduce_sum(a, name='sum_c')

        with tf.name_scope('output'):
            output = tf.add(b, c, name='output')

    with tf.name_scope('update'):
        # accumulate the output overtime
        update_total = total_output.assign_add(output)
        # increment step by 1
        increment_step = global_step.assign_add(1)

    with tf.name_scope('summaries'):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name='average')

        tf.summary.scalar('output_summary', output)
        tf.summary.scalar('total_summary', total_output)
        tf.summary.scalar('average_summary', avg)

    with tf.name_scope('global_ops'):
        init = tf.global_variables_initializer()
        merged_summaries = tf.summary.merge_all()


def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)


# running the graph
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter('./my_graph/computation_graph/summary', graph)
    sess.run(init)
    for _ in range(0, 10):
        input_shape = np.random.randint(1, 3)
        input_tensor = np.random.randn(2)
        run_graph(input_tensor)
    writer.flush()
    writer.close()
