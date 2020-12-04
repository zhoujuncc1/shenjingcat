# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
from tensorflow.keras.utils import to_categorical
sess=tf.InteractiveSession()


num_classes = 2


def CNNnet(inputs,n_class):

    conv1 = tf.layers.conv1d(inputs=inputs, filters=16, kernel_size=3,strides=1, \
                             padding='same', activation = tf.nn.relu,use_bias = False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv1)

    avg_pool_1 = tf.layers.average_pooling1d(inputs=conv1, pool_size=2, strides=2, \
                                         padding='same')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, avg_pool_1)

    conv2 = tf.layers.conv1d(inputs=avg_pool_1, filters=24, kernel_size=3, strides=1,\
                             padding='same', activation = tf.nn.relu,use_bias = False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv2)
    avg_pool_2 = tf.layers.average_pooling1d(inputs=conv2, pool_size=2, strides=2,\
                                         padding='same')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, avg_pool_2)
    """
    conv3 = tf.layers.conv1d(inputs=avg_pool_2, filters=32, kernel_size=7, strides=1, \
                             padding='same', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv3)
    
    avg_pool_3 = tf.layers.average_pooling1d(inputs=conv3, pool_size=2, strides=2, \
                                             padding='same')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, avg_pool_3)
    
    conv4 = tf.layers.conv1d(inputs=avg_pool_3, filters=16, kernel_size=5, strides=1, \
                             padding='same', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv4)

    avg_pool_4 = tf.layers.average_pooling1d(inputs=conv4, pool_size=2, strides=2, \
                                             padding='same')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, avg_pool_4)

    conv5 = tf.layers.conv1d(inputs=avg_pool_4, filters=32, kernel_size=3, strides=1, \
                             padding='same', activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv4)

    avg_pool_5 = tf.layers.average_pooling1d(inputs=conv5, pool_size=2, strides=2, \
                                             padding='same')
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, avg_pool_5)
    """
    flat = tf.reshape(avg_pool_2, (-1, 45 * 32))

    #dense1 =tf.layers.dense(inputs=flat, units=64, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    #tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, dense1)
    #dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
    #dense3 = tf.layers.dense(inputs=dense2, units=128, activation=tf.nn.relu)
    #dense4 = tf.layers.dense(inputs=dense3, units=64, activation=tf.nn.relu)
    logits=tf.layers.dense(inputs=flat, units=2, activation=tf.nn.softmax, use_bias=False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, logits)
    #logits=tf.nn.relu(logits)
    #tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, logits)
    return logits


def get_batch(train_x,train_y,batch_size, index):
    #indices=np.random.choice(train_x.shape[0],batch_size,False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.05))
    if index == 549:#549
        end =  len(train_x)
    else:
        end = (index+1)*batch_size
    batch_x=train_x[index*batch_size:end]
    batch_y=train_y[index*batch_size:end]
    return batch_x,batch_y


def getresultwvlt():
    print("Loading data and labels...")
    tic=time.time()


    print("Divide training and testing set...")
    #path = './all_0719_paper_smoke_01.npz'
    #path = './all_0810_paper_smoke_normalize_01.npz'
    #path = './raw_0805_paper_re_normalize_01.npz'
    #path = './raw_0714_paper_extend_01.npz'
    #path = './all_0801_paper_smoke_01.npz'
    #path = './all_0807_paper_smoke_normalize_.npz'
    path = './all_0810_paper_smoke_normalize_01.npz'
    f = np.load(path)
    train_x, train_y = f['x_train'], f['y_train']
    test_x, test_y = f['x_test'], f['y_test']
    
    #print(test_x.shape)
    #print(test_y.shape)
    #print(train_x.shape)
    #print(train_y.shape)
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y.shape)
    toc=time.time()
    print("Elapsed time is %f sec."%(toc-tic))
    print("======================================")


    print("1D-CNN setup and initialize...")
    tic=time.time()
    x=tf.placeholder(tf.float32, [None, 180], name='input')
    x_=tf.reshape(x,[-1,180,1])
    y_=tf.placeholder(tf.float32,[None,2], name='output')

    logits=CNNnet(x_,2)

    learning_rate=0.001
    batch_size=500
    maxiters=5000


    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
    train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.global_variables_initializer().run()
    toc=time.time()



    print("Elapsed time is %f sec."%(toc-tic))
    print("======================================")

    print("1D-CNN training and testing...")
    tic=time.time()
    #saver = tf.train.Saver()
    #saver.restore(sess, "./0810_180_03/model.ckpt")
       
    for i in range(maxiters):
        #saver.restore(sess, "./model.ckpt")
        batch_x,batch_y=get_batch(train_x,train_y,batch_size, i % 550)#550
        train_step.run(feed_dict={x:batch_x,y_:batch_y})
        if i%550==0:#550
            loss=cost.eval(feed_dict={x:batch_x,y_:batch_y})
            print("Iteration %d/%d:loss %f"%(i,maxiters,loss))
    
    saver = tf.train.Saver()
    saver.save(sess, "./2020_180_01/model.ckpt")
    
    #sess = tf.Session()
    #print(sess.run(logits))
    #y_pred_ = logits.eval(feed_dict={x:train_x})
    #print(y_pred_)
    y_pred=logits.eval(feed_dict={x:test_x})
    y_pred=np.argmax(y_pred,axis=1)
    y_true=np.argmax(test_y,axis=1)
    toc=time.time()
    print(y_pred)
    print(y_true)

    Acc=np.mean(y_pred==y_true)
    Conf_Mat=confusion_matrix(y_true,y_pred)
    Acc_N=Conf_Mat[0][0]/np.sum(Conf_Mat[0])
    Acc_V=Conf_Mat[1][1]/np.sum(Conf_Mat[1])
    #Acc_R=Conf_Mat[2][2]/np.sum(Conf_Mat[2])
    #Acc_L=Conf_Mat[3][3]/np.sum(Conf_Mat[3])

    print(Acc_N)
    print('\nAccuracy=%.2f%%'%(Acc*100))
    print('Accuracy_N=%.2f%%'%(Acc_N*100))
    print('Accuracy_V=%.2f%%'%(Acc_V*100))
    #print('Accuracy_R=%.2f%%'%(Acc_R*100))
    #print('Accuracy_L=%.2f%%'%(Acc_L*100))
    print('\nConfusion Matrix:\n')
    print(Conf_Mat)
    print(np.sum(Conf_Mat[0]))
    print("======================================raw")

    return y_pred

y= getresultwvlt()
#print (y)
#np.savez('2class_0715.npz',y_pred=y)
