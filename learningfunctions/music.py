#!/usr/local/bin/python
import numpy as np
import tensorflow as tf
import glob
import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

"""
def cnn_model_fn(features, labels, mode):
Model function for CNN.
############### Input Layer #########################
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) #remodelisation de l entree
#####################################################

"""

path = '../projet/tensorflow-music-generator/Pop_Music_Midi/'
def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_to_statematrix.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
        return songs

# HyperParameters
# First, let's take a look at the hyperparameters of our model:

lowest_note = 24  # the index of the lowest note on the piano roll
highest_note = 102  # the index of the highest note on the piano roll
note_range = highest_note - lowest_note  # the note range

num_timesteps = 15  # This is the number of timesteps that we will create at a time
n_visible = 2 * note_range * num_timesteps  # This is the size of the visible layer.
n_hidden = 50  # This is the size of the hidden layer

num_epochs = 200  # The number of training epochs that we are going to run.
# For each epoch we go through the entire data set.
batch_size = 100  # The number of training examples that we are going to send through the RBM at a time.
lr = tf.constant(0.005, tf.float32)  # The learning rate of our model

# Variables:
# Next, let's look at the variables we're going to use:
# The placeholder variable that holds our data
x = tf.placeholder(tf.float32, [None, n_visible], name="x")
# The weight matrix that stores the edge weights
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
# The bias vector for the hidden layer
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))
# The bias vector for the visible layer
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))


with tf.session as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for epoch in tqdm(range(num_epochs)):
        for song in songs:            
