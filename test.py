import scipy
from scipy import ndimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict, compute_cost, initialize_parameters

def get_parameters():
    """
    get parameters form saved session
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('trained_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
        
    ### START CODE HERE ### (approx. 6 lines of code)
    #W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b1 = tf.get_variable('b1', [25,1], initializer = tf.zeros_initializer())
    #W2 = tf.get_variable('W2', [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b2 = tf.get_variable('b2', [12,1], initializer = tf.zeros_initializer())
    #W3 = tf.get_variable('W3', [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #b3 = tf.get_variable('b3', [6,1], initializer = tf.zeros_initializer())

    #print all tensor name
    graph = tf.get_default_graph()
    #print([n.name for n in graph.as_graph_def().node])    
    W1 = graph.get_tensor_by_name("W1:0")
    b1 = graph.get_tensor_by_name("b1:0")
    W2 = graph.get_tensor_by_name("W2:0")
    b2 = graph.get_tensor_by_name("b2:0")
    W3 = graph.get_tensor_by_name("W3:0")
    b3 = graph.get_tensor_by_name("b3:0")
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

if __name__ == '__main__':
    my_image = "thumbs_up.jpg"
    
    # We preprocess your image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    
    parameters = get_parameters()
    my_image_prediction = predict(my_image, parameters)
    
    plt.imshow(image)
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
