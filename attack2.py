#-*-coding:utf-8-*- 
__author__ = 'guojunfeng'
import numpy as np
import pandas as pd
from keras import *
import tensorflow as tf
import keras.datasets.mnist as mnist
from setup_mnist import MNISTModel

#get batch
def batch_loader(train_data,train_label):

    train_list=[]
    label_list=[]
    mat=np.eye((10))

    for i in range(10):


        for h in range(9):
          train_list.append(train_data[train_label==i][0])

        for m in range(10):
            if (m==i):
                continue
            label_list.append(mat[m])

    return(np.array(train_list),np.array(label_list))





#load mnist data
def load_data():
    (train_data,train_label),(test_data,test_label)=mnist.load_data()
    return((train_data.reshape(train_data.shape[0],28,28,1))/255,train_label,test_data/255,test_label)

"""
attack2 where original_img is origianl image from keras.mnist target_vector is target num's vector

"""
def attack2(original_img,target_vector,batch,model,sess):

    #label is used to contain loss associated with c
    #label=[]



    img_size=original_img.shape[1]
    channel_num=original_img.shape[3]

    #shape of our input
    shape=(batch,channel_num,channel_num,1)

    #attack_pixel is the VAriable we want get through backpropgation

    attack_pixel=tf.Variable(np.zeros((90,28,28,1),np.float32))

    img=tf.Variable(np.zeros((batch,img_size,img_size,channel_num),np.float32))
    vector=tf.Variable(np.zeros((batch,10),np.float32),tf.float32)
    c=tf.Variable(np.zeros((batch,1),np.float32),tf.float32)
    new_img=0.5*tf.tanh(attack_pixel+img)+0.5

    output=model.predict(new_img)

    assign_img=tf.placeholder (shape=(90, 28, 28, 1), dtype=tf.float32)
    assign_vector=tf.placeholder (shape=(batch, 10), dtype=tf.float32)
    assign_c=tf.placeholder (shape=(batch, 1), dtype=tf.float32)

    dis_loss_ditribute=tf.reduce_sum(tf.square(new_img-tf.tanh(img)*0.5-0.5),[1,2,3])

    non_target_v=tf.reduce_max(output*(1-vector)-10000*vector,1)
    target_v=tf.reduce_sum(vector*output,1)

    dis_loss=tf.reduce_sum(dis_loss_ditribute)

    loss=c*tf.reduce_sum(tf.maximum(0.0,non_target_v-target_v))+dis_loss



    start_vars = set (x.name for x in tf.global_variables ())

    op = tf.train.AdamOptimizer (0.01)
    train = op.minimize (loss, var_list=[attack_pixel])

    end_vars = tf.global_variables ()

    new_vars = [x for x in end_vars if x.name not in start_vars]

    init_var=tf.variables_initializer(var_list=[attack_pixel]+new_vars)

    sess.run(init_var)
    c_initial=np.array([0.1]*90).reshape(90,1)
    #c_step=1000
    c_step=1
    const_c=np.array([1.00]*90).reshape(90,1)
    label_acc=[]
    label_disloss=[]
    upper_band=np.array([150.0]*90)
    lower_band=np.array([0.0]*90)
    # just begin with our session
    for step in range(c_step):
        # initialize the lowest distance when attack successfully
      #lowest_dist = np.array ([1000] * batch)

        # initialize best constant value
      #best_constant = np.array ([2] * batch)

      #const_c=c_initial+step*0.1

      #const_c=np.array([0.01*np.power(10,step)]*90).reshape(90,1)

      sess.run([img.assign(assign_img),vector.assign(assign_vector),c.assign(assign_c)],{assign_img:original_img,
                        assign_vector:target_vector,
                        assign_c:const_c}
               )
    #train
      for i in range(10000):
        sess.run(train)
        #loss_instant=sess.run(dis_loss_ditribute)
        #print(loss_instant)
        #index=(np.argmax (sess.run (model.predict(new_img)), 1)== np.argmax (label_batch,1)) & (loss_instant<=lowest_dist)
        #print(index)
      instant_output=(np.argmax(sess.run(model.predict(new_img)),1)==np.argmax(label_batch,1))
      print(sum(instant_output)/90)
      for k in range(len(instant_output)):
        if instant_output[k]==False:

            lower_band[k]=const_c[k]
            const_c[k]=(lower_band[k]+upper_band[k])/2

        if instant_output[k]==True:

            upper_band[k]=const_c[k]
            const_c[k]=(upper_band[k]+lower_band[k])/2

      print(max(lower_band),max(upper_band))







        #for h in range(len(index)):

            #if index[h]==True:
                #lowest_dist[h]=loss_instant[h]
                #print(h%10)
                #print(sess.run(model.predict(new_img)[h]))
                #print(np.argmax(label_batch,1)[h])
                #print(loss_instant[h])
                #print(np.argmax(sess.run(model.predict(tf.tanh(img)*0.5+0.5))[h]))
                #print(sess.run(attack_pixel[h]))




      #label_acc.append(sum(np.argmax(sess.run(model.predict(new_img)),1)==np.argmax(label_batch,1))/90)
      #label_disloss.append(sum(lowest_dist[lowest_dist<1000])/len(lowest_dist<1000))
      #print(lowest_dist)
    #print(label_disloss)
    #print(label_acc)
    #print(label_disloss)
    print(sum(const_c)/90,min(const_c),max(const_c))


#load data

train_data,train_label,test_data,test_label=load_data()

data_batch,label_batch=batch_loader(train_data,train_label)

#load model
with tf.Session() as sess:

 model=MNISTModel('models/mnist',sess)

 original_img=np.arctanh((data_batch-0.5)*2*0.999)

 print(attack2(data_batch,label_batch,90,model,sess))
























