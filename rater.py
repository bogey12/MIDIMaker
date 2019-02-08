import data1
import tensorflow as tf
import midi
import numpy as np
'''test with mnist dimension network'''

lowerBound = 24
upperBound = 102

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(
        x, W, strides=[1, 1, 1, 1],
        padding='SAME'
    )


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME'
    )

def train_rater():
    with tf.Session() as sess:
        with tf.variable_scope('rater'):
            a = Rater()
            a.full_build(a.x)
            sess.run(tf.global_variables_initializer())
            total_score = 0
            steps = 0
            for i in range(10):
                batch = data1.next_batch()
                train_accuracy = a.error.eval(feed_dict={
                    a.x: batch[1], a.y_: batch[0], a.keep_prob: 1.0})
                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                #if i % 5 == 0:
                  #  saver.save(sess, "saved_models/rater.ckpt")
                a.train_step.run(feed_dict={a.x: batch[1], a.y_: batch[0], a.keep_prob: 0.5})
                print('test accuracy %g' % a.error.eval(feed_dict={
                    a.x: batch[1], a.y_: batch[0], a.keep_prob: 1.0}))
                grads_and_vars = tf.train.AdamOptimizer(1e-4).compute_gradients(a.error)
                for gv in grads_and_vars:
                    print str(gv[0]) + " - " + gv[1].name
            a.saver.save(sess, "saved_models/rater.ckpt")
            # print ('cross entropy is %d' % cross_entropy.eval(feed_dict={
                # x: batch[1], y_: batch[0], keep_prob: 1.0}))
def train_composer1():
    with tf.Session() as sess:
        with tf.variable_scope('composer'):
            b = Composer()
            sess.run(tf.global_variables_initializer())
            steps = 0
            for i in range(10):
                batch = data1.next_batch()
                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                #if i % 5 == 0:
                  #  saver.save(sess, "saved_models/rater.ckpt")
                b.train_step.run(feed_dict={b.x: batch[1], b.keep_prob: 0.5})
                print('test accuracy %g' % b.error.eval(feed_dict={
                    b.x: batch[1], b.keep_prob: 1.0}))
                grads_and_vars = tf.train.AdamOptimizer(1e-4).compute_gradients(b.error)
            for gv in grads_and_vars:
               print str(gv[0]) + " - " + gv[1].name
            b.saver.save(sess, "saved_models/composer.ckpt")

def train_composer2():
    with tf.Session() as sess:
        #with tf.variable_scope('rater'):
        a = Rater()
        #a.saver.restore(sess, 'saved_models/rater.ckpt')
        #with tf.variable_scope('composer'):
        # b = Composer()
            #b.saver.restore(sess, "saved_models/composer.ckpt")
        # 

        sess.run(tf.global_variables_initializer())


        grads_and_vars = tf.train.AdamOptimizer(1e-4).compute_gradients(a.error)
        for gv in grads_and_vars:
            print str(gv[0]) + " - " + gv[1].name
        #train_step2 = tf.train.AdamOptimizer(1e-4).minimize(a.error2)
        grads_and_vars = tf.train.AdamOptimizer(1e-4).compute_gradients(a.error2, var_list = a.tvars)
        for gv in grads_and_vars:
            print str(gv[0]) + " - " + gv[1].name

        
        
        g = 0
        #print a.W_fc2.eval()
        print a.W_fc2.eval()

        '''Harmony Copy'''

        for c in range(3000):
            try:
                batch1 = data1.next_batch('Harmony', 1)

                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                #if i % 5 == 0:
                  #  saver.save(sess, "saved_models/rater.ckpt")
                a.train_step.run(feed_dict={a.z_h: batch1[1], a.x2: batch1[1], a.keep_prob: 0.5, 
                    a.y_: batch1[0], a.x: batch1[1]})
                if c % 50 == 0:
                    print 'step %d' % c
                print('test accuracy %g' % a.error.eval(feed_dict={
                    a.z_h: batch1[1], a.x2: batch1[1], a.keep_prob: 1.0, a.y_: batch1[0],
                    a.x: batch1[1]}))
            except Exception as e:
                print(e)
                continue

        '''Rater Train'''
        for i in range(3000):
            try:
                batch1 = data1.next_batch('output', 1)

                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                #if i % 5 == 0:
                  #  saver.save(sess, "saved_models/rater.ckpt")
                a.train_step.run(feed_dict={a.x: batch1[1], a.y_: batch1[0], a.keep_prob: 0.5})
                if i % 50 == 0:
                    print 'step %d' % i
                print('test accuracy %g' % a.error.eval(feed_dict={
                    a.x: batch1[1], a.y_: batch1[0], a.keep_prob: 1.0}))
            except Exception as e:
                print(e)
                continue

        


        # a.full_build(b.y_conv
        '''Composer Train'''
        for p in range(3000):

            batch = data1.next_batch('music', 1)
                #print('step %d, training accuracy %g' % (i, train_accuracy))
            #if i % 5 == 0:
              #  saver.save(sess, "saved_models/rater.ckpt")
            try:
                a.train_step3.run(feed_dict={a.x2: batch[1], a.keep_prob: 0.5})
                if p % 50 == 0:
                    print 'step %d' % p
                print('test accuracy %g' % a.error3.eval(feed_dict={
                    a.x2: batch[1], a.keep_prob: 1.0}))
            except Exception as e:
                print(e)
                continue
        
        '''Harmony Train'''
        '''
        for h in range(5000):
            try:
                batchh = data1.next_batch('output', 1)

                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                #if i % 5 == 0:
                  #  saver.save(sess, "saved_models/rater.ckpt")
                a.train_step.run(feed_dict={a.xh: batchh[1], a.y_h: batchh[0], a.keep_prob: 0.5})
                if h % 50 == 0:
                    print 'step %d' % h
                print('test accuracy %g' % a.error.eval(feed_dict={
                    a.xh: batchh[1], a.y_h: batchh[0], a.keep_prob: 1.0}))
            except Exception as e:
                print(e)
                continue
                '''

       #  inputi = a.y_conv.eval(feed_dict={a.x2: batch[1], a.keep_prob2: 0.5})
        '''Combined + Generation'''
        for b in range(5000): 
            try:
                batch = data1.next_batch('music', 1)
                a.train_step2.run(feed_dict={a.x2: batch[1], a.keep_prob: 0.5})
                # inputi = a.y_conv.eval(feed_dict={a.x: inputi, a.keep_prob: 0.5})
                #b = tf.convert_to_tensor(batch[1], dtype=tf.float32)
                if b % 50 == 0:
                    print 'step %d' % b
                print('test accuracy %g' % a.error3.eval(feed_dict={
                    a.x2: batch[1], a.keep_prob: 1.0}))
            except Exception as e:
                print(e)
                continue
            if b % 50 == 0:

                batch = data1.next_batch('music', 1)
                for y in range(3):
                    try:
                        matrix = a.y_conv2.eval(feed_dict={a.x2: batch[1], a.keep_prob: 0.5})
                        '''for x in np.nditer(matrix, flags=['buffered'], op_flags = ['readwrite']):
                                prob = x * 100
                                roll = np.random.randint(100)
                                chance = np.random.randint(1000)
                                if prob < roll:
                                    x[...] = 0
                                else:
                                    if  prob < chance:
                                        x[...] = 0
                                    else:
                                        x[...] = 1'''

                        matrix = np.resize(matrix, (200*36*2))
                        
                        '''for c in range(1000):
                            matrix1 = a.y_conv2.eval(feed_dict={a.x2: batch[1], a.keep_prob: 0.5})[2*78*28:]
                            matrix = np.append(matrix, matrix1)
                            del matrix1'''
                        try:
                            matrix = np.reshape(matrix, (-1, 36, 2))
                            #print matrix
                            data1.noteStateMatrixToMidi(matrix, 'output/tediuasesisesesisesesisese'+str(g))
                        except Exception as e:
                            print(e)
                            continue
                        g += 1
                    except Exception as e:
                        print(e)
                        continue
        



class Rater():
    def __init__(self):
        # with tf.variable_scope('rater'):
        '''composer network'''
        self.x2 = tf.placeholder(tf.float32, shape=[None, 112*36], name='x2')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # self.y_ = tf.placeholder(tf.float32, shape=[None, 78*500])

        self.W_conv12 = weight_variable([3, 3, 1, 64])
        self.b_conv12 = bias_variable([64])

        self.W_conv32 = weight_variable([3, 3, 64, 64])
        self.b_conv32 = bias_variable([64])

        self.W_conv52 = weight_variable([3, 3, 64, 64])
        self.b_conv52 = bias_variable([64])

        '''self.W_conv62 = weight_variable([3, 3, 64, 64])
        self.b_conv62 = bias_variable([64])

        self.W_conv72 = weight_variable([3, 3, 64, 128])
        self.b_conv72 = bias_variable([128])

        self.W_conv82 = weight_variable([3, 3, 128, 128])
        self.b_conv82 = bias_variable([128])

        self.W_conv92 = weight_variable([3, 3, 128, 128])
        self.b_conv92 = bias_variable([128])

        self.W_conv102 = weight_variable([3, 3, 128, 128])
        self.b_conv102 = bias_variable([128])'''

        self.x_image2 = tf.reshape(self.x2, [-1, 112, 36, 1])

        self.h_conv12 = tf.nn.sigmoid(conv2d(self.x_image2, self.W_conv12) + self.b_conv12)
        self.h_conv32 = tf.nn.sigmoid(conv2d(self.h_conv12, self.W_conv32) + self.b_conv32)
        self.h_conv52 = tf.nn.sigmoid(conv2d(self.h_conv32, self.W_conv52) + self.b_conv52)
        #self.h_conv62 = tf.nn.sigmoid(conv2d(self.h_conv52, self.W_conv62) + self.b_conv62)

        '''self.h_conv72 = tf.nn.sigmoid(conv2d(self.h_conv62, self.W_conv72) + self.b_conv72)
        self.h_conv82 = tf.nn.sigmoid(conv2d(self.h_conv72, self.W_conv82) + self.b_conv82)
        self.h_conv92 = tf.nn.sigmoid(conv2d(self.h_conv82, self.W_conv92) + self.b_conv92)
        self.h_conv102 = tf.nn.sigmoid(conv2d(self.h_conv92, self.W_conv102) + self.b_conv102)'''
        self.h_pool12 = max_pool_2x2(self.h_conv52)

        '''Harmony Combined'''

        self.xh = tf.placeholder(tf.float32, shape=[None, 112*36], name='xh')
        self.y_h = tf.placeholder(tf.float32, shape=[None, 1])

        '''For harmony compy'''
        self.z_h = tf.placeholder(tf.float32, shape=[None, 112*36], name='z_h')

        self.W_fcx1 = weight_variable([16128, 112*36])
        self.b_fcx1 = bias_variable([112*36])

        self.h_pool12_flat = tf.reshape(self.h_pool12, [-1, 16128])


        self.h_fcx1 = tf.nn.relu(tf.matmul(self.h_pool12_flat, self.W_fcx1) + self.b_fcx1)
        #Comparison to h_fcx1


        self.x_image3 = tf.reshape(self.h_fcx1, [-1, 112, 36, 1])
        self.xh_image = tf.reshape(self.xh, [-1, 112, 36, 1])


        self.W_convh1 = weight_variable([3, 3, 1, 64])
        self.b_convh1 = bias_variable([64])

        self.W_convh2 = weight_variable([3, 3, 64, 128])
        self.b_convh2 = bias_variable([128])

        self.W_convh3 = weight_variable([3, 3, 128, 256])
        self.b_convh3 = bias_variable([256])


        self.hc_1 = tf.nn.relu(conv2d(self.x_image3, self.W_convh1) + self.b_convh1)
        self.hc_2 = tf.nn.relu(conv2d(self.hc_1, self.W_convh2) + self.b_convh2)
        self.hc_3 = tf.nn.relu(conv2d(self.hc_2, self.W_convh3) + self.b_convh3)
        self.hc_pool1 = max_pool_2x2(self.hc_3)

        self.h_1 = tf.nn.relu(conv2d(self.xh_image, self.W_convh1) + self.b_convh1)
        self.h_2 = tf.nn.relu(conv2d(self.h_1, self.W_convh2) + self.b_convh2)
        self.h_3 = tf.nn.relu(conv2d(self.h_2, self.W_convh3) + self.b_convh3)
        self.h_pool17 = max_pool_2x2(self.h_3)


        self.W_fcx2 = weight_variable([16128, 112*36])
        self.b_fcx2 = bias_variable([112*36])


        self.hc_pool1_flat = tf.reshape(self.hc_pool1, [-1, 16128])
        self.h_fcx2 = tf.nn.relu(tf.matmul(self.hc_pool1_flat, self.W_fcx2) + self.b_fcx2)
        self.h_fcx1_drop = tf.nn.dropout(self.h_fcx2, self.keep_prob)

        self.h_pool17_flat = tf.reshape(self.h_pool17, [-1, 16128])
        self.h_fcx27 = tf.nn.relu(tf.matmul(self.h_pool17_flat, self.W_fcx2) + self.b_fcx2)
        self.h_fcx17_drop = tf.nn.dropout(self.h_fcx27, self.keep_prob)


        self.W_fcx3 = weight_variable([112*36, 1])
        self.b_fcx3 = bias_variable([1])


        self.yhc_conv = tf.matmul(self.h_fcx1_drop, self.W_fcx3) + self.b_fcx3
        self.yh_conv = tf.matmul(self.h_fcx17_drop, self.W_fcx3) + self.b_fcx3

        self.thvars = [self.W_fcx1, self.b_fcx1, self.W_conv12, self.b_conv12, self.W_conv32,
            self.b_conv32, self.W_conv52, self.b_conv52]


        '''Harmony Combined End'''
        self.W_conv22 = weight_variable([3, 3, 1, 64])
        self.b_conv22 = bias_variable([64])

        self.W_conv42 = weight_variable([3, 3, 64, 128])
        self.b_conv42 = bias_variable([128])

        self.W_conv112 = weight_variable([3, 3, 128, 256])
        self.b_conv112 = bias_variable([256])

        '''self.W_conv122 = weight_variable([3, 3, 256, 256])
        self.b_conv122 = bias_variable([256])

        self.W_conv132 = weight_variable([3, 3, 256, 512])
        self.b_conv132 = bias_variable([512])

        self.W_conv142 = weight_variable([3, 3, 512, 512])
        self.b_conv142 = bias_variable([512])

        self.W_conv152 = weight_variable([3, 3, 512, 512])
        self.b_conv152 = bias_variable([512])

        self.W_conv162 = weight_variable([3, 3, 512, 512])
        self.b_conv162 = bias_variable([512])'''
                #
        self.h_conv22 = tf.nn.relu(conv2d(self.x_image3, self.W_conv22) + self.b_conv22)
        self.h_conv42 = tf.nn.relu(conv2d(self.h_conv22, self.W_conv42) + self.b_conv42)
        self.h_conv112 = tf.nn.relu(conv2d(self.h_conv42, self.W_conv112) + self.b_conv112)
        '''self.h_conv122 = tf.nn.relu(conv2d(self.h_conv112, self.W_conv122) + self.b_conv122)

        self.h_conv132 = tf.nn.relu(conv2d(self.h_conv122, self.W_conv132) + self.b_conv132)
        self.h_conv142 = tf.nn.relu(conv2d(self.h_conv132, self.W_conv142) + self.b_conv142)
        self.h_conv152 = tf.nn.relu(conv2d(self.h_conv142, self.W_conv152) + self.b_conv152)
        self.h_conv162 = tf.nn.relu(conv2d(self.h_conv152, self.W_conv162) + self.b_conv162)'''
        self.h_pool22 = max_pool_2x2(self.h_conv112)

        self.W_fc12 = weight_variable([16128, 4096])
        self.b_fc12 = bias_variable([4096])

        self.h_pool2_flat2 = tf.reshape(self.h_pool22, [-1, 16128])
        self.h_fc12 = tf.nn.relu(tf.matmul(self.h_pool2_flat2, self.W_fc12) + self.b_fc12)

        #self.W_fc32 = weight_variable([2048, 112*76*2])
        #self.b_fc32 = bias_variable([112*76*2])

        #self.h_fc22 = tf.nn.relu(tf.matmul(self.h_fc12, self.W_fc32) + self.b_fc32)

        
        self.h_fc1_drop2 = tf.nn.dropout(self.h_fc12, self.keep_prob)

        self.W_fc22 = weight_variable([4096, 112*36])
        self.b_fc22 = bias_variable([112*36])

        self.y_conv2 = tf.nn.sigmoid(tf.matmul(self.h_fc1_drop2, self.W_fc22) + self.b_fc22)

        '''self.tvars = [self.W_conv12, self.b_conv12, self.W_conv22, self.b_conv22, self.W_conv32, self.b_conv32,
            self.W_conv42, self.b_conv42, self.W_conv52, self.b_conv52, self.W_conv62, self.b_conv62, self.W_conv72,
            self.b_conv72, self.W_conv82, self.b_conv82, self.W_conv92, self.b_conv92, self.W_conv102, self.b_conv102,
            self.W_conv112, self.b_conv112, self.W_conv122, self.b_conv122, self.W_conv132, self.b_conv132, self.W_conv142,
            self.b_conv142, self.W_conv152, self.b_conv152, self.W_conv162, self.b_conv162, self.W_fc12,
            self.b_fc12, self.W_fc22, self.b_fc22]'''

        self.t1vars = [self.W_conv22, self.b_conv22, self.W_conv42, self.b_conv42, self.W_conv112,
            self.b_conv112, self.W_fc12, self.b_fc12, self.W_fc22, self.b_fc22]

        self.tvars = [self.W_conv12, self.b_conv12, self.W_conv22, self.b_conv22, self.W_conv32, self.b_conv32,
            self.W_conv42, self.b_conv42, self.W_conv52, self.b_conv52,
            self.W_conv112, self.b_conv112, self.W_fc12,
            self.b_fc12, self.W_fc22, self.b_fc22]

        '''rater network'''

        self.x = tf.placeholder(tf.float32, shape=[None, 112*36], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

        self.W_conv1 = weight_variable([3, 3, 1, 64])
        self.b_conv1 = bias_variable([64])

        self.W_conv3 = weight_variable([3, 3, 64, 64])
        self.b_conv3 = bias_variable([64])

        self.W_conv5 = weight_variable([3, 3, 64, 64])
        self.b_conv5 = bias_variable([64])

        '''self.W_conv6 = weight_variable([3, 3, 64, 64])
        self.b_conv6 = bias_variable([64])

        self.W_conv7 = weight_variable([3, 3, 64, 128])
        self.b_conv7 = bias_variable([128])

        self.W_conv8 = weight_variable([3, 3, 128, 128])
        self.b_conv8 = bias_variable([128])

        self.W_conv9 = weight_variable([3, 3, 128, 128])
        self.b_conv9 = bias_variable([128])

        self.W_conv10 = weight_variable([3, 3, 128, 128])
        self.b_conv10 = bias_variable([128])'''

        self.x_image = tf.reshape(self.x, [-1, 112, 36, 1])

        self.h_conv1 = tf.nn.sigmoid(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_conv3 = tf.nn.sigmoid(conv2d(self.h_conv1, self.W_conv3) + self.b_conv3)
        self.h_conv5 = tf.nn.sigmoid(conv2d(self.h_conv3, self.W_conv5) + self.b_conv5)
        '''self.h_conv6 = tf.nn.sigmoid(conv2d(self.h_conv5, self.W_conv6) + self.b_conv6)

        self.h_conv7 = tf.nn.sigmoid(conv2d(self.h_conv6, self.W_conv7) + self.b_conv7)
        self.h_conv8 = tf.nn.sigmoid(conv2d(self.h_conv7, self.W_conv8) + self.b_conv8)
        self.h_conv9 = tf.nn.sigmoid(conv2d(self.h_conv8, self.W_conv9) + self.b_conv9)
        self.h_conv10 = tf.nn.sigmoid(conv2d(self.h_conv9, self.W_conv10) + self.b_conv10)'''
        self.h_pool1 = max_pool_2x2(self.h_conv5)

        self.W_conv2 = weight_variable([3, 3, 64, 128])
        self.b_conv2 = bias_variable([128])

        self.W_conv4 = weight_variable([3, 3, 128, 256])
        self.b_conv4 = bias_variable([256])

        self.W_conv11 = weight_variable([3, 3, 256, 256])
        self.b_conv11 = bias_variable([256])

        '''self.W_conv12 = weight_variable([3, 3, 256, 256])
        self.b_conv12 = bias_variable([256])

        self.W_conv13 = weight_variable([3, 3, 256, 512])
        self.b_conv13 = bias_variable([512])

        self.W_conv14 = weight_variable([3, 3, 512, 512])
        self.b_conv14 = bias_variable([512])

        self.W_conv15 = weight_variable([3, 3, 512, 512])
        self.b_conv15 = bias_variable([512])

        self.W_conv16 = weight_variable([3, 3, 512, 512])
        self.b_conv16 = bias_variable([512])'''
                #
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_conv4 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv4) + self.b_conv4)
        self.h_conv11 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv11) + self.b_conv11)
        '''self.h_conv12 = tf.nn.relu(conv2d(self.h_conv11, self.W_conv12) + self.b_conv12)

        self.h_conv13 = tf.nn.relu(conv2d(self.h_conv12, self.W_conv13) + self.b_conv13)
        self.h_conv14 = tf.nn.relu(conv2d(self.h_conv13, self.W_conv14) + self.b_conv14)
        self.h_conv15 = tf.nn.relu(conv2d(self.h_conv14, self.W_conv15) + self.b_conv15)
        self.h_conv16 = tf.nn.relu(conv2d(self.h_conv15, self.W_conv16) + self.b_conv16)'''
        self.h_pool2 = max_pool_2x2(self.h_conv11)

        self.W_fc1 = weight_variable([16128, 4096])
        self.b_fc1 = bias_variable([4096])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 16128])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        #self.W_fc3 = weight_variable([2048, 112*76*2])
        #self.b_fc3 = bias_variable([112*76*2])

        #self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc3) + self.b_fc3)

        #self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([4096, 1])
        self.b_fc2 = bias_variable([1])

        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2



        '''combine layers'''

        self.x_image1 = tf.reshape(self.y_conv2, [-1, 112, 36, 1])
        self.h_conv1a = tf.nn.sigmoid(conv2d(self.x_image1, self.W_conv1) + self.b_conv1)
        self.h_conv3a = tf.nn.sigmoid(conv2d(self.h_conv1a, self.W_conv3) + self.b_conv3)
        self.h_conv5a = tf.nn.sigmoid(conv2d(self.h_conv3a, self.W_conv5) + self.b_conv5)
        '''self.h_conv6a = tf.nn.sigmoid(conv2d(self.h_conv5a, self.W_conv6) + self.b_conv6)

        self.h_conv7a = tf.nn.sigmoid(conv2d(self.h_conv6a, self.W_conv7) + self.b_conv7)
        self.h_conv8a = tf.nn.sigmoid(conv2d(self.h_conv7a, self.W_conv8) + self.b_conv8)
        self.h_conv9a = tf.nn.sigmoid(conv2d(self.h_conv8a, self.W_conv9) + self.b_conv9)
        self.h_conv10a = tf.nn.sigmoid(conv2d(self.h_conv9a, self.W_conv10) + self.b_conv10)'''
        self.h_pool1a = max_pool_2x2(self.h_conv5a)

        self.h_conv2a = tf.nn.relu(conv2d(self.h_pool1a, self.W_conv2) + self.b_conv2)
        self.h_conv4a = tf.nn.relu(conv2d(self.h_conv2a, self.W_conv4) + self.b_conv4)
        self.h_conv11a = tf.nn.relu(conv2d(self.h_conv4a, self.W_conv11) + self.b_conv11)
        '''self.h_conv12a = tf.nn.relu(conv2d(self.h_conv11a, self.W_conv12) + self.b_conv12)

        self.h_conv13a = tf.nn.relu(conv2d(self.h_conv12a, self.W_conv13) + self.b_conv13)
        self.h_conv14a = tf.nn.relu(conv2d(self.h_conv13a, self.W_conv14) + self.b_conv14)
        self.h_conv15a = tf.nn.relu(conv2d(self.h_conv14a, self.W_conv15) + self.b_conv15)
        self.h_conv16a = tf.nn.relu(conv2d(self.h_conv15a, self.W_conv16) + self.b_conv16)'''
        self.h_pool2a = max_pool_2x2(self.h_conv11a)

        self.h_pool2_flata = tf.reshape(self.h_pool2a, [-1, 16128])
        self.h_fc1a = tf.nn.relu(tf.matmul(self.h_pool2_flata, self.W_fc1) + self.b_fc1)
        #self.h_fc2a = tf.nn.relu(tf.matmul(self.h_fc1a, self.W_fc3) + self.b_fc3)
        self.h_fc1_dropa = tf.nn.dropout(self.h_fc1a, self.keep_prob)
        self.y_conva = tf.matmul(self.h_fc1_dropa, self.W_fc2) + self.b_fc2



        '''combine training'''
        self.error2 = tf.negative(self.y_conva)
        self.train_step2 = tf.train.AdamOptimizer(1e-4).minimize(self.error2, var_list = self.t1vars)

        '''composer training'''
        self.error3 = tf.losses.mean_squared_error(self.x2, self.y_conv2)
        self.train_step3 = tf.train.AdamOptimizer(1e-4).minimize(self.error3, var_list = self.t1vars)

        '''rater training'''
        self.error = tf.reduce_mean(tf.abs(tf.subtract(self.y_conv,self.y_)))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.error)
        '''self.vars = [self.W_conv1, self.b_conv1, 
            self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3,
            self.W_conv4, self.b_conv4, self.W_conv5, self.b_conv5,
            self.W_conv11, self.b_conv11, self.W_fc1, self.b_fc1, self.W_fc2, 
            self.b_fc2]'''

        '''Harymony Copy training'''
        self.error6 = tf.losses.mean_squared_error(self.z_h, self.h_fcx1)
        self.train_step6 = tf.train.AdamOptimizer(1e-4).minimize(self.error6)

        '''Harmony combined training'''
        self.error4 = tf.negative(self.y_conva)
        self.train_step4 = tf.train.AdamOptimizer(1e-4).minimize(self.error4, var_list = self.thvars)
        

        '''Harmony training'''
        self.error5 = tf.reduce_mean(tf.abs(tf.subtract(self.yh_conv,self.y_h)))
        self.train_step5 = tf.train.AdamOptimizer(1e-4).minimize(self.error5)
        
        self.saver = tf.train.Saver()
    
if __name__ == '__main__':
    #train_rater()
    #train_composer1()
    train_composer2()
    #data1.generate()
