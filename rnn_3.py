import tensorflow as tf
#
# x = [[1, 2, 3],
#      [1, 2, 3]]
#
# xx = tf.cast(x, tf.float32)
#
# mean_all = tf.reduce_mean(xx, keep_dims=False)
# mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
# mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)
#
# with tf.Session() as sess:
#     m_a, m_0, m_1 = sess.run([mean_all, mean_0, mean_1])
#
# print(m_a)  # output: 2.0
# print(m_0)  # output: [ 1.  2.  3.]
# print(m_1)  # output:  [ 2.  2.]
print('ok')