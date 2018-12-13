import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

class_data = list()
final_prob = 0.0
final_class = ''
image_class = dict()

def predict_image(image_path):
    image_class['class'] = 'None'
    image_class['prob'] = 0.0
    # First, pass the path of the image
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # image_path = 'output/test/index.jpeg' #sys.argv[1]
    filename = dir_path +'/' + image_path
    image_size =128
    num_channels =3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    try:
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)

    except Exception as e:
        print((str(e)))

    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('parts-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir('output/train'))))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]

    with open('classes.txt', 'r') as f:
        class_data = f.read().splitlines()

    final_prob = 0
    final_class = 'None'

    for i in range((result.size)):
        # print(class_data[i], "--->", float("{0:.6f}".format(result[0,i])))
        if (result[0,i] > final_prob):
            final_prob = result[0,i]
            final_class = class_data[i]

    image_class['class'] = final_class
    image_class['prob'] = str(final_prob)
    print (image_class)
    return image_class


if __name__ == '__main__':
    predict_image('input.png')