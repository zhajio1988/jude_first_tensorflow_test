import scipy
from scipy import ndimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict, compute_cost, initialize_parameters
from model import ckpt_dir 

def get_parameters():
    """
    get parameters form saved session
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1) #so that your "random" numbers match ours
    sess = tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(ckpt_dir + '/trained_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    #print all tensor name
    #print([n.name for n in graph.as_graph_def().node])    
    #extract parameters from saved session
    W1 = sess.run("W1:0")
    b1 = sess.run("b1:0")
    W2 = sess.run("W2:0")
    b2 = sess.run("b2:0")
    W3 = sess.run("W3:0")
    b3 = sess.run("b3:0")    

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters, sess

if __name__ == '__main__':
    my_image = "thumbs_up.jpg"
    
    # We preprocess your image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    
    parameters, sess= get_parameters()
    my_image_prediction = predict(my_image, parameters, sess)
    
    plt.imshow(image)
    plt.show()
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
