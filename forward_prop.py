import numpy as np
import tensorflow as tf

#def forward_propagation(X, parameters, keep_prob, train):
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    

    def batch_norm_layer(x, train_phase, scope_bn):
        with tf.variable_scope(scope_bn):
            beta = tf.get_variable("beta", [x.shape[-1]], initializer = tf.zeros_initializer(),trainable=True)
            gamma = tf.get_variable("gamma", [x.shape[-1]], initializer = tf.ones_initializer(),trainable=True)
            axises = np.arange(len(x.shape) - 1)
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments', keep_dims=True)
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(train_phase, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))

            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
    

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    #Z1 = tf.matmul(W1,X) 
    #Z1 = batch_norm_layer(tf.transpose(Z1), train, 'BN1')
    #Z1 = tf.nn.dropout(tf.transpose(Z1), keep_prob) 
    #A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    ##Z2 = tf.add(tf.matmul(W2,A1), b2)
    #Z2 = tf.matmul(W2,A1)
    #Z2 = batch_norm_layer(tf.transpose(Z2),train,'BN2')
    #Z2 = tf.nn.dropout(tf.transpose(Z2), keep_prob)
    #A2 = tf.nn.relu(Z2)             
    ##Z3 = tf.add(tf.matmul(W3,A2), b3)# A2 = relu(Z2)
    #Z3 = tf.matmul(W3,A2)
    #Z3 = batch_norm_layer(tf.transpose(Z3),train,'BN3')
    #Z3 = tf.nn.dropout(tf.transpose(Z3), keep_prob)
    ### END CODE HERE ###
    
    return Z3
