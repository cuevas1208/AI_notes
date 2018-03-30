'''
Manuel Cuevas
Term3, project road segmentation

To run this code you would need a GPU with at least 6GB of memory, because my did not have one with me I used floyhub.

To run the project under floydhub type this line in your command line:
floyd run --gpu --env tensorflow-1.3 --data cuevas1208/datasets/data_road/1:/data_road
--data cuevas1208/datasets/pretrained_vgg/1:/pretrained_vgg "python main.py"

'''
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
# import tqdm

# That disables use of scratch memory in your model
# TF_CUDNN_WORKSPACE_LIMIT_IN_MB=0

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class FCN(object):

    def __init__(self, parameters):
        """
        Load settings parameters
        :param params:
        """
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])


    def load_vgg(self, sess, vgg_path):
        """
        Load Pretrained VGG Model into TensorFlow.
        :param sess: TensorFlow Session
        :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
        :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
        """
        # Use tf.saved_model.loader.load to load the model and weights
        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        graph = tf.get_default_graph()

        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        layer3_out = graph.get_tensor_by_name('layer3_out:0')
        layer4_out = graph.get_tensor_by_name('layer4_out:0')
        layer7_out = graph.get_tensor_by_name('layer7_out:0')

        return image_input, keep_prob, layer3_out, layer4_out, layer7_out


    def layers(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
        """
        Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
        :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
        :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
        :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
        :param num_classes: Number of classes to classify
        :return: The Tensor for the last layer of output
        """
        #kernel_initializer = tf.truncated_normal_initializer(stddev=self.init_sd)
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)

        # 1x1 convolutions of the three layers
        conv_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                  kernel_regularizer=kernel_regularizer)
        conv_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                  kernel_regularizer=kernel_regularizer)
        conv_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                  kernel_regularizer=kernel_regularizer)

        # Upsample layer 7 and add to layer 4
        # tf.layers.conv2d_transpose(inputs,filters,kernel_size,strides=(1, 1), padding='valid'...)
        upsample_1 = tf.layers.conv2d_transpose(conv_7, num_classes, 4, 2, 'SAME',
                                           kernel_regularizer=kernel_regularizer)
        upsample_1 = tf.add(upsample_1, conv_4)

        # Upsample the sum and add to layer 3
        upsample_2 = tf.layers.conv2d_transpose(upsample_1, num_classes, 4, 2, 'SAME',
                                           kernel_regularizer=kernel_regularizer)
        upsample_2 = tf.add(upsample_2, conv_3)

        # Upsample the input and return
        upsample_3 = tf.layers.conv2d_transpose(upsample_2, num_classes, 16, 8, 'SAME',
                                           kernel_regularizer=kernel_regularizer)
        return upsample_3


    def optimize(self, nn_last_layer, correct_label, learning_rate, num_classes):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the correct label image
        :param learning_rate: TF Placeholder for the learning rate
        :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """
        # Reshape logits for computing cross entropy
        logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')

        # Compute cross entropy and loss
        cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)

        # All regularization terms are added to a collection called tf.GraphKeys.REGULARIZATION_LOSSES,
        # add the sum of all regularization losses to the previously calculated cross-entropy
        cross_entropy_loss = tf.reduce_mean(cross_entropy_logits) + \
                             sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Training operation using the Adam optimizer
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

        return logits, train_op, cross_entropy_loss


    def train_nn(self, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate):
        """
        Train neural network and print out the loss during training.
        :param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
        :param train_op: TF Operation to train the neural network
        :param cross_entropy_loss: TF Tensor for the amount of loss
        :param input_image: TF Placeholder for input images
        :param correct_label: TF Placeholder for label images
        :param keep_prob: TF Placeholder for dropout keep probability
        :param learning_rate: TF Placeholder for learning rate
        """
        # Iterate over epochs
        for epoch in range(1, epochs + 1):
            print("Epoch: " + str(epoch) + "/" + str(epochs))

            # Iterate over the batches using the batch generation function
            total_loss = []
            loss = 0
            batch = get_batches_fn(batch_size)
            size = int((self.training_images / batch_size)+.5)

            for d in batch:
                # Create the feed dictionary
                image, label = d
                feed_dict = {
                    input_image: image,
                    correct_label: label,
                    keep_prob: self.dropout,
                    learning_rate: self.learning_rate
                }

                # Train and compute the loss
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
                total_loss.append(loss)

            # Compute mean epoch loss
            mean_loss = sum(total_loss) / size
            print("Loss:  " + str(loss) + "\n")

    def save_model(self, sess):
        """
        :param sess:
        :return:
        """
        saver = tf.train.Saver()
        saver.save(sess, self.save_location)
        tf.train.write_graph(sess.graph_def, self.save_location, "saved_model.pb", False)

    def run_tests(self):
        """
        :return:
        """
        tests.test_load_vgg(self.load_vgg, tf)
        tests.test_layers(self.layers)
        tests.test_optimize(self.optimize)
        tests.test_train_nn(self.train_nn)
        tests.test_for_kitti_dataset(self.data_dir)

    def run(self):
        """
        OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
        You'll need a GPU with at least 10 teraFLOPS to train on.
        https://www.cityscapes-dataset.com/

        :return:
        """
        # check compatibility and maybe download pre-trained vgg model
        helper.maybe_download_pretrained_vgg(self.vgg_path)

        with tf.Session() as sess:
            # Path to vgg model
            vgg_path = os.path.join(self.vgg_path, 'vgg')

            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(self.data_dir, self.training_subdir),
                                                       self.image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # Placeholders
            learning_rate = tf.placeholder(dtype=tf.float32)
            correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, self.num_classes))

            # Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, l3, l4, l7 = self.load_vgg(sess, vgg_path)
            output = self.layers(l3, l4, l7, self.num_classes)
            logits, train_op, cross_entropy_loss = self.optimize(output, correct_label, learning_rate,
                                                                               self.num_classes)
            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Train NN using the train_nn function
            self.train_nn(sess, self.epochs, self.batch_size, get_batches_fn, train_op, cross_entropy_loss,
                          input_image, correct_label, keep_prob, learning_rate)

            # Save inference data using helper.save_inference_samples
            helper.save_inference_samples(self.runs_dir, os.path.join(self.data_dir, self.testing_subdir),
                                          sess, self.image_shape, logits, keep_prob, input_image)

            # OPTIONAL: Apply the trained model to a video

            # Save the model
            self.save_model(sess)


if __name__ == '__main__':
    parameters = {
        'learning_rate':   0.0001,
        'dropout':         0.5,
        'epochs':          100,
        'batch_size':      10,
        'init_sd':         0.01,
        'training_images': 289,
        'num_classes':     2,
        'image_shape':     (160, 576),
        'training_subdir': 'training',
        'testing_subdir':  'testing',

        # set up for local run
        #'data_dir':        '/tmp/datasets/data_road',
        #'vgg_path':        '/tmp/models/pretrained_vgg',
        #'save_location':   './output/',
        #'runs_dir':        './ output / runs /'

        # set up to run in on  Floyhub
        'data_dir':        '/data_road',
        'vgg_path':        '/pretrained_vgg',
        'save_location':   '/output/',
        'runs_dir':        '/output/runs/'

    }

    fcn = FCN(parameters)
    fcn.run_tests()
    fcn.run()