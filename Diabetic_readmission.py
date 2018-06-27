#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:07:27 2018

@author: deepakmishra
"""

# Diabetes Readmission
import tensorflow as tf
import os

# Assuming 15 features -- Gender, Age, Admission type, Discharge disposition
# Admission source, Time in hospital, Number of lab procedures
# Number of procedures, Number of medications, Number of outpatient visits
# Number of emergency visits, Number of inpatient visits, Number of diagnoses
# Change of medications & Readmitted
W = tf.Variable(tf.zeros([12,1]), name ="weights") 
b = tf.Variable(0., name="bias")

def combine_inputs(X):
    return tf.matmul(X,W) + b

def inference(X):
    return tf.sigmoid(combine_inputs(X))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([
            os.path.join(os.getcwd(), file_name)])
    
    reader = tf.TextLineReader(skip_header_lines = 1)
    key, value = reader.read(filename_queue)
    
    decoded = tf.decode_csv(value, record_defaults = record_defaults)
    
    return tf.train.shuffle_batch(decoded,
                                  batch_size = batch_size,
                                  capacity = batch_size*1050,
                                  min_after_dequeue = batch_size)

    
default_record =  [[0.0], [0.0],[""], [""], [""], [""], 
                   [0.0],[0.0], [0.0], 
                   [0.0], [0.0], [0.0], [0.0],
                   [0.0], [0.0], [0.0],[""],[""],
                   [""], [0.0],[""],[""],[""],[0.0]]
    
def inputs():
    encounter_id,patient_nbr,race,gender,age,weight,\
    admission_type_id,discharge_disposition_id,admission_source_id,\
    time_in_hospital,num_lab_procedures,num_procedures,num_medications,\
    number_outpatient,number_emergency,number_inpatient,diag_1,diag_2,\
    diag_3,number_diagnoses,max_glu_serum,change,diabetesMed,readmitted = \
    read_csv(100, "diabetic_data.csv",default_record)

 ## Convert Catagorical data - Weight not part of the features
    
    which_gender = tf.to_float(tf.equal(gender, ["Female"]))
    is_med_change = tf.to_float(tf.equal(change, ["Ch"]))
    is_diabetesMed = tf.to_float(tf.equal(diabetesMed, ["Yes"]))
    
   # features = tf.transpose(tf.stack([which_gender,admission_type_id,discharge_disposition_id,admission_source_id,
   #                                  time_in_hospital,num_lab_procedures,num_procedures,num_medications,
   #                                  number_outpatient,number_emergency,number_inpatient,
   #                                  number_diagnoses,is_med_change,is_diabetesMed]))
    
    features = tf.transpose(tf.stack([which_gender,admission_type_id,discharge_disposition_id,admission_source_id,
                                     time_in_hospital,num_lab_procedures,num_procedures,
                                     number_emergency,number_inpatient, number_diagnoses,is_med_change,is_diabetesMed]))
    
    
    readmitted = tf.reshape(readmitted, [100,1])

    
    return features, readmitted

        
#Using tarining we adjust the model parameters
def train(total_loss) :
    learning_rate = 0.005
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


#evaluate the resulting model
def evaluate(sess, X, Y):
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print("Accuracy of Readmission Prediction-->", sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted,Y), tf.float32)))*100, "%")


def loss(X,Y):
    
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))
    

    
# Launch graph and run the training loop
    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    X, Y = inputs()
    total_loss = loss(X,Y)
    train_op = train(total_loss)
    
    # define a coordinator to start and stop the threads
    coord = tf.train.Coordinator()
    # wake up the threads
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #Actual Training loop
    training_steps = 5000
    for step in range(training_steps):
        
        sess.run([train_op])
        #See how the loss gets decremented through training steps
        if step % 1000 == 0:
            print("Epoch:", step, " loss", sess.run(total_loss))
    
    print("Final model W =", sess.run(W), "b=", sess.run(b))
    evaluate(sess, X, Y)
    #import time
    #time.sleep(5)
    
    # When done, ask the threads to stop.
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    
    sess.close()