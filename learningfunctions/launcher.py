################################################
###              LAUNCHER class              ###
### launch the Training / Musique Generation ###
################################################

class Launcher:
    def __init__(self):
        print('Init')
        #init Self params
        self.n_visible = None
        self.n_hidden = None
        self.num_epochs = None
        self.batch_size = None
        self.lr = None
        self.x = None
        self.W = None
        self.bh = None
        self.bv = None
        self.logits = None
        self.loss = None

    def launch():

        ################################################
    	###    	fonction convertion midi to matrix   ###
    	################################################
    	
            
        ######################################
    	###    Hyperparameters             ###
    	######################################
        lowest_note = 24
        highest_note = 102
        note_range = highest_note - lowest_note

        num_timesteps = 15
        self.n_visible = 2 * note_range * num_timesteps
        self.n_hidden = 50

        self.num_epochs = 200
        self.batch_size = 100
        
        self.lr = tf.constant(0.005, tf.float32)
            
        ######################################
    	###    	variables                  ###
    	######################################
        self.x = tf.placeholder(tf.float32, [None, self.n_visible], name='x')
        self.W = tf.Variable(tf.random_normal([self.n_visible, self.n_hidden], 0.01), name='W')
        self.bh = tf.Variable(tf.zeros([1, self.n_hidden], tf.float32, name='bh'))
        self.bv = tf.Variable(tf.zeros([1, self.n_visible], tf.float32, name='bv'))
            
    	######################################
    	###    	 fonction de creation	   ###
    	###    des couches de neuronnes    ###
    	######################################
        self.setLogits()

    	######################################
    	###    	fonction de cout           ###
    	######################################
        self.costFunction()

    	###########################################
    	###   fonction de descente de gradient  ###
    	###########################################

    	
    	
    	######################################
    	###    	fonction de training       ###
    	######################################
    	
    	
    	######################################
    	###    	fonction d evaluation      ###
    	######################################



    def setLogits():
        with tf.name_scope('dnn'):
            hidden1 = tf.layers.dense(self.x, self.n_hidden, name='hidden', activation=tf.nn.relu)
            #hidden2 = tf.layers.dense(hidden1, self.n_hidden2, name='hidden2', activation = tf.nn.relu)
            #hidden3 = tf.layers.dense(hidden2, self.n_hidden3, name='hidden3', activation = tf.nn.relu)
            self.logits = tf.layers.dense(hidden1, self.n_visible, name='outputs')


    def costFunction():
        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.x, logits=self.logits)
            self.loss = tf.reduce_mean(xentropy, name='loss')
