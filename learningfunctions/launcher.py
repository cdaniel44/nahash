#Launcher class => in this class, we'll launch he training

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import import_midi


class Launcher:

    def __init__(self):
        print('Init')
        #init Self params


    def launch():
        #TODO All the entry point
        
        ######################################
	###          songs                 ###
	######################################

        songs = import_midi.get_songs("../musics/")
        x_test = songs ##################################################################### a redimensionner
        

        ######################################
	###    Hypermarameters             ###
	######################################
	
        lowest_note = import_midi.lowerBound  # the index of the lowest note on the piano roll
        highest_note = import_midi.upperBound  # the index of the highest note on the piano roll
        note_range = highest_note - lowest_note  # the note range

        n_steps = 15  # This is the number of timesteps that we will create at a time
        n_inputs = 2 * note_range * n_steps  # This is the size of the visible layer.
        n_neurons = 150  # This is the size of the hidden layer
        n_outputs = 10

        n_epochs = 100  # The number of training epochs that we are going to run.
        # For each epoch we go through the entire data set.
        batch_size = 10  # The number of training examples that we are going to send through the RNN at a time.
        n_batches = len(songs) // batch_size
        learning_rate = tf.constant(0.005, tf.float32)  # The learning rate of our model
        learning_rate = 0.001
        
    ######################################
	###    	variables                  ###
	######################################
        

        
        
        x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        y = tf.placeholder(tf.int32, [None])
        

	######################################
	###    	fonction de creation	   ###
	###     des couches de neuronnes   ###
	######################################

        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)
        
        logits = tf.layers.dense(states, n_outputs)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        
        

	######################################
	###    	fonction de perte          ###
	######################################

        loss = tf.reduce_mean(xentropy)
        
        
	###########################################
	###   fonction d optimisation           ###
	###########################################
	
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

	###########################################
	###   fonction de descente de gradient  ###
	###########################################
	
	
	##########################################
	### fonction de training /evaluation   ###
	##########################################
	
        training_op = optimizer.minimize(loss)
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	######################################
	###    	        Run                ###
	######################################
	
        init = tf.global_variables_initializer()

        with.tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for iteration in range(n_batches):
                    x_batch, y_batch =

    
    
    
    if __name__ == "__main__":
        launch()
    
    
    
    
    
    
    
    
    
