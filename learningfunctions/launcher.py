import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import import_midi
import midi_manipulation
from tensorflow.python.ops import control_flow_ops


class Launcher:

    def __init__(self):
        print('Init')
        #init Self params

    def launch():
        #TODO All the entry point

    ######################################
	###    Hypermarameters             ###
	######################################

        lowest_note = import_midi.lowerBound
        highest_note = import_midi.upperBound
        note_range = highest_note - lowest_note
        truncated_backprop_length = 15
        n_inputs = 2 * note_range * truncated_backprop_length
        num_steps = 15
        num_neurons = 150
        num_outputs = n_inputs

        num_epochs = 100
        batch_size = 10
        state_size = 4
        num_classes = 2
        songs = import_midi.get_songs("../musics/")
        songs = np.array(songs)
        print(songs.shape)
        num_batches = len(songs) // batch_size
        learning_rate = tf.constant(0.005, tf.float32)

    ######################################
	###    	variables                  ###
	######################################

        x = tf.placeholder(tf.float32, [None, None, n_inputs])
        y = tf.placeholder(tf.int32, [None, None])
        seqlen = tf.placeholder

        W = tf.Variable(np.random.rand(batch_size, n_inputs, batch_size), dtype=tf.float32)
        b = tf.Variable(np.zeros((1, batch_size)), dtype=tf.float32)

        W2 = tf.Variable(np.random.rand(batch_size, n_inputs, batch_size),dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1, batch_size)), dtype=tf.float32)

        cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons)
        outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        logits_series = tf.layers.dense(outputs, num_outputs)


	######################################
	###    	fonction de perte          ###
	######################################

        print(logits_series)
        print(y)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_series)
        total_loss = tf.reduce_mean(losses)

	###########################################
	###   fonction d optimisation           ###
	###########################################

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_step = optimizer.minimize(total_loss)

	###########################################
	###   fonction de descente de gradient  ###
	###########################################

        def sample(probs):
            return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

        def gibbs_sample(k):
            def gibbs_step(count, k, xk):
                return count + 1, k, xk

            ct = tf.constant(0)
            [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter, gibbs_step, [ct, tf.constant(k), x])
            x_sample = tf.stop_gradient(x_sample)
            return x_sample

	######################################
	###    	        Run                ###
	######################################

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            updt = [total_loss, train_step, states]

            for epoch_idx in range(num_epochs):
                _states = np.zeros((batch_size, state_size))

                print("New song, epoch :", epoch_idx)
                print(len(songs))
                for song in songs:
                    if len(song) > 1 :
                        song = np.array(song)
                        song = song[:int(np.floor(song.shape[0] // num_steps) * num_steps)]
                        print(song)
                        print(song.shape)
                        song = np.reshape(song, [-1, song.shape[0] // num_steps, song.shape[1] * num_steps])
                        for i in range(1, len(song), batch_size):
                            tr_x = song[i:i + batch_size]
                            print(x.shape)
                            print(tr_x.shape)
                            print(tr_x)
                            _total_loss, _train_step, _states = sess.run(updt, feed_dict={ x: tr_x })

            sample = gibbs_sample(1).eval(session=sess, feed_dict={ x: np.zeros((batch_size, num_steps, n_inputs)) })
            for i in range(sample.shape[0]):
                smpl = np.array(sample[i, :])
                spml = smpl[:int(np.floor(smpl.shape[0] // num_steps) * num_steps)]
                print(smpl)
                S = np.reshape(smpl, [-1, smpl.shape[0] // num_steps, smpl.shape[1] * num_steps])
                midi_manipulation.noteStateMatrixToMidi(S, "../out/generated_chord_{}".format(i))


    if __name__ == "__main__":
        launch()
