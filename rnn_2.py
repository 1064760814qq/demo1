# 定义RNN网络
import urllib.request
def RNN(X, weights, biases):
    # inputs=[batch_size,max_time,n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])  # 50*784  =》50*28*28
    # 定义LSTM基本CELL   基础的LSTM循环网络单元
    # __init__(

    #     num_units,
    #
    #     forget_bias=1.0,
    #
    #     state_is_tuple=True,
    #
    #     activation=None,
    #
    #     reuse=None,
    #
    #     name=None,
    #
    #     dtype=None
    #
    # )
    #     num_units: int类型，LSTM单元中的神经元数量，即输出神经元数量
    #     forget_bias: float类型，偏置增加了忘记门。从CudnnLSTM训练的检查点(checkpoin)
    #     恢复时，必须手动设置为0.0。
    #     state_is_tuple: 如果为True，则接受和返回的状态是c_state和m_state的2 - tuple；如果为False，则他们沿着列轴连接。后一种即将被弃用。
    #     activation: 内部状态的激活函数。默认为tanh
    #     reuse: 布尔类型，描述是否在现有范围中重用变量。如果不为True，并且现有范围已经具有给定变量，则会引发错误。
    #     name: String类型，层的名称。具有相同名称的层将共享权重，但为了避免错误，在这种情况下需要reuse = True.
    #     dtype: 该层默认的数据类型。默认值为None表示使用第一个输入的类型。在call之前build被调用则需要该参数。
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)  # n_neurons=lstm_size=100
    # final_state[0]是cell state
    # final_state[1]是hidden_state 与最后的输出有关

    # tf.nn.dynamic_rnn(
    #     cell,
    #     inputs,
    #     sequence_length=None,
    #     initial_state=None,
    #     dtype=None,
    #     parallel_iterations=None,
    #     swap_memory=False,
    #     time_major=False,
    #     scope=None
    # )

    # inputs:[50*28*28]即[batch_size,step,input_size]
    # 按照规定，outputs是最后一层的输出，即为[batch_size,step,n_neurons]n_neurons是神经元的个数
    # 按照规定，final_state是每一层的最后一个step的输出，其实本程序只是用了一个隐藏层，因为是LSTM长短时记忆网络，所以因此我们的states包含1个LSTMStateTuple，
    # 每一个表示每一层的最后一个step的输出，这个输出有两个信息，一个是h表示短期记忆信息，一个是c表示长期记忆信息。尺寸为[batch_size,n_neurons]
    # 所以，ouputs是[50,28,100]，final_state每个LSTMStateTuple都包含c,h两个矩阵，都是[50*100]
    # 给大家举例看一个形状：这里的[batch_size,step,n_neurons]是[4,2,5]   h c的尺寸分别是[4,5]
    # outputs_val:
    # [[[1.2949290e-04 0.0000000e+00 2.7623639e-04 0.0000000e+00 0.0000000e+00]
    #   [9.4675866e-05 0.0000000e+00 2.0214770e-04 0.0000000e+00 0.0000000e+00]]
    #
    # [[4.3100454e-06 4.2123037e-07 1.4312843e-06 0.0000000e+00 0.0000000e+00]
    # [0.0000000e+00
    # 0.0000000e+00
    # 0.0000000e+00
    # 0.0000000e+00
    # 0.0000000e+00]]
    #
    # [[0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
    #  [0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]]
    #
    # [[0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
    #  [0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]]]
    #
    # states_val:
    # (LSTMStateTuple(
    #     c=array([[0., 0., 0.04676079, 0.04284539, 0.],
    #              [0., 0., 0.0115245, 0., 0.],
    #              [0., 0., 0., 0., 0.],
    #              [0., 0., 0., 0., 0.]],
    #             dtype=float32),
    #     h=array([[0., 0., 0.00035096, 0.04284406, 0.],
    #              [0., 0., 0.00142574, 0., 0.],
    #              [0., 0., 0., 0., 0.],
    #              [0., 0., 0., 0., 0.]],
    #             dtype=float32)),
    #  LSTMStateTuple(
    #      c=array([[0.0000000e+00, 1.0477135e-02, 4.9871090e-03, 8.2785974e-04,
    #                0.0000000e+00],
    #               [0.0000000e+00, 2.3306280e-04, 0.0000000e+00, 9.9445322e-05,
    #                5.9535629e-05],
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                0.0000000e+00],
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                0.0000000e+00]], dtype=float32),
    #      h=array([[0.00000000e+00, 5.23016974e-03, 2.47756205e-03, 4.11730434e-04,
    #                0.00000000e+00],
    #               [0.00000000e+00, 1.16522635e-04, 0.00000000e+00, 4.97301044e-05,
    #                2.97713632e-05],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #                0.00000000e+00],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #                0.00000000e+00]], dtype=float32)),
    #  LSTMStateTuple(
    #      c=array([[1.8937115e-04, 0.0000000e+00, 4.0442235e-04, 0.0000000e+00,
    #                0.0000000e+00],
    #               [8.6200516e-06, 8.4243663e-07, 2.8625946e-06, 0.0000000e+00,
    #                0.0000000e+00],
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                0.0000000e+00],
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                0.0000000e+00]], dtype=float32),
    #      h=array([[9.4675866e-05, 0.0000000e+00, 2.0214770e-04, 0.0000000e+00,
    #                0.0000000e+00],
    #               [4.3100454e-06, 4.2123037e-07, 1.4312843e-06, 0.0000000e+00,
    #                0.0000000e+00],
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                0.0000000e+00],
    #               [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
    #                0.0000000e+00]], dtype=float32))
    #  - --------------------
    #  版权声明：本文为CSDN博主「小T是我」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
    # 原文链接：https: // blog.csdn.net / junjun150013652 / article / details / 81331448
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    # 所以final_state[1]代表的就是LSTMStateTuple里的h,final_state[0]代表的是c
    results = tf.matmul(final_state[1], weights) + biases  # 返回[50,10]
    return results