
#tuto : https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

###############################################
# fonctions tensorflow

# tf.concat : Concatenates tensors along one dimension

# tf.reshape : redimensionne un tensor ex :
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
#reshape(t, [3, 3]) ==> [[1, 2, 3],
#                        [4, 5, 6],
#                        [7, 8, 9]]

# tf.unstack decoupe un tableau à plusieurs dimensions en plusieurs tableau à x dimension
# exemple :
# var2 = tf.Variable([[1,2,3],[4,5,6]],tf.int32)
# var3 = tf.unstack(var2)
# var2: [[1 2 3][4 5 6]]
# var3: [array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32)]

#################################################


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
        num_steps = 15
        truncated_backprop_length = 15  # This is the number of timesteps that we will create at a time
        num_inputs = 2 * note_range * truncated_backprop_length  # This is the size of the visible layer.
        num_neurons = 150  # This is the size of the hidden layer
        num_outputs = 10

        num_epochs = 100  # The number of training epochs that we are going to run.
        # For each epoch we go through the entire data set.
        batch_size = 10  # The number of training examples that we are going to send through the RNN at a time.
        state_size = 4
        num_classes = 2
        num_batches = len(songs) // batch_size
        learning_rate = tf.constant(0.005, tf.float32)  # The learning rate of our model
        
    ######################################
	###    	variables                  ###
	######################################
        

        
        #placeholder that holds data inputs 
        x = tf.placeholder(tf.float32, [None, truncated_backprop_length, n_inputs])
        y = tf.placeholder(tf.int32, [None, truncated_backprop_length, n_inputs])
        init_state = tf.placeholder(tf.float32, [batch_size, state_size])
        
        
        W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
        b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

        W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
        
        inputs_series = tf.unstack(x, axis=1)
        labels_series = tf.unstack(y, axis=1)
        
        #build RNN that does the actual RNN computation
        
        current_state = init_state
        states_series = []
        for current_input in inputs_series:
            current_input = tf.reshape(current_input, [batch_size, 1])
            input_and_state_concatenated = tf.concat(1, [current_input, current_state]) # Increasing number of columns
            next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b) # Broadcasted addition
            states_series.append(next_state)
            current_state = next_state
            
        logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
        predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
        
        
	######################################
	###    	fonction de creation	   ###
	###     des couches de neuronnes   ###
	######################################



	######################################
	###    	fonction de perte          ###
	######################################

        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
        total_loss = tf.reduce_mean(losses)


        
	###########################################
	###   fonction d optimisation           ###
	###########################################
	
        train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


	###########################################
	###   fonction de descente de gradient  ###
	###########################################
	
	
	##############################################
	### fonction de visualisation du training  ###
	##############################################
	
        
        def plot(loss_list, predictions_series, batchX, batchY):
            plt.subplot(2, 3, 1)
            plt.cla()
            plt.plot(loss_list)

            for batch_series_idx in range(5):
                one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
                single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

                plt.subplot(2, 3, batch_series_idx + 2)
                plt.cla()
                plt.axis([0, truncated_backprop_length, 0, 2])
                left_offset = range(truncated_backprop_length)
                plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
                plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
                plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

            plt.draw()
            plt.pause(0.0001)

	######################################
	###    	        Run                ###
	######################################
	

        with.tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            plt.ion()
            plt.figure()
            plt.show()
            loss_list = []
            
            for epoch_idx in tqdm(range(num_epochs)):
                for iteration in range(n_batches):
                    songs = np.array(songs)
                    song = song[:int(np.floor(song.shape[0] // n_steps) * n_steps)]
                    song = np.reshape(song, [song.shape[0] // n_steps, song.shape[1] * n_steps])
                    for i in range(1, len(song), batch_size):
                        tr_x = song[i:i + batch_size]
                        sess.run(updt, feed_dict={x: tr_x})

    
    


    
    if __name__ == "__main__":
        launch()
    
    











    
