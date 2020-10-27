import tensorflow as tf
#from tensorflow.contrib import rnn
#import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#载入数据集
mnist = input_data.read_data_sets("F:/Pycharm/pycharm-workspace/tensorflow_learn/MNISTdata",one_hot = True)

#输入图片是28*28。第一次输入28个数据，一共输入28次。
n_inputs = 28                        #输入一行，一行有28个数据
max_time = 28                        #一共28行
lstm_size = 100                      #隐层单元
n_classes = 10                       #10个分类
batch_size = 50                      #每个批次50个样本
n_batch = mnist.train.num_examples   #计算一共有多少批次
print(n_batch)
#这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32,[None,784])                   #50行，784列
#正确的标签
y = tf.placeholder(tf.float32,[None,10])

#初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev = 0.1))      #相当于[100,10]
#初始化偏置值
biases = tf.Variable(tf.constant(0.1,shape = [n_classes]))

#定义RNN函数
def RNN(X,weights,biases):
    #inputs = [batch_size,max_time,n_inputs]
    inputs = tf.reshape(X,[-1,max_time,n_inputs])               #转换X格式，[50,784]变为[50,28,28]
    #定义LSTM基本CELL
    lstm_cell = lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    print(lstm_cell)
    #final_state[0]是cell_state
    #final_state[1]是hidden_state
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype = tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
    return results

#计算RNN的返回结果
prediction = RNN(x,weights,biases)
# print(prediction.shape())
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))

#使用AdamOptimizer优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)         #优化器，学习率可改。。学习率为10的负4次方。用AdamOptimizer一般将学习率调的较小，0.01算较大的

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
#比较equal中两个参数是否一样，一样则返回True
#tf.argmax()返回一维张量中最大值所在的位置。（求最大概率的数字在第几个位置，即标签）
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率。tf.cast()用来将布尔类型转换为float32位浮点型的
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):                                                                 #轮数可改
        for batch in range(n_batch):                                                        #批次循环
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})            #可改。。keep_prob:1.0所有神经元都工作

        #准确率变化
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        #打印周期和准确率
        print("Iter " + str(epoch) + ",  Testing Accuracy " + str(acc))