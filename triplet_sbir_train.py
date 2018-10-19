import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import spatial_softmax
from sbir_sampling import triplet_sampler_asy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

net_id = 0  # 0 for step3 pre-trained model, 1 for step2 pre-trained model


def attentionNet(inputs, pool_method='sigmoid'):
    assert(pool_method in ['sigmoid', 'softmax'])
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=True):
        net = slim.conv2d(inputs, 256, [1, 1], padding='SAME', scope='conv1')
        if pool_method == 'sigmoid':
            net = slim.conv2d(net, 1, [1, 1], activation_fn=tf.nn.sigmoid, scope='conv2')
        else:
            net = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv2')
            net = spatial_softmax(net)
    return net


def sketch_a_net_sbir(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')  # trainable=trainable
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            conv5 = slim.flatten(conv5)
            fc6 = slim.fully_connected(conv5, 512, trainable=trainable, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=trainable, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
    return fc7


def sketch_a_net_dssa(inputs, trainable):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):  # when test 'trainable=True', don't forget to change it
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', trainable=trainable, scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            # residual attention
            att_mask = attentionNet(conv5, 'softmax')
            att_map = tf.multiply(conv5, att_mask)
            att_f = tf.add(conv5, att_map)
            attended_map = tf.reduce_sum(att_f, reduction_indices=[1, 2])
            attended_map = tf.nn.l2_normalize(attended_map, dim=1)
            att_f = slim.flatten(att_f)
            fc6 = slim.fully_connected(att_f, 512, trainable=trainable, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, trainable=trainable, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
            # coarse-fine fusion
            final_feature_map = tf.concat([fc7, attended_map], 1)
    return final_feature_map


def init_variables(model_file):
    # d = np.load(model_file, encoding="latin1").item()
    # pretrained_paras = d.keys()
    #
    # init_ops = []  # a list of operations
    # for var in tf.global_variables():
    #     for w_name in pretrained_paras:
    #         if w_name in var.name:
    #             init_ops.append(var.assign(d[w_name]))

    if net_id == 0:
        pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1', 'fc7_sketch']
    else:
        pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1']
    d = np.load(model_file, encoding="latin1").item()
    init_ops = []  # a list of operations
    for var in tf.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                print('Initialise var %s with weight %s' % (var.name, w_name))
                try:
                    if 'weights' in var.name:
                        init_ops.append(var.assign(d[w_name]['weights']))
                    elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name]['biases']))
                except KeyError:
                     if 'weights' in var.name:
                        init_ops.append(var.assign(d[w_name+'/weights:0']))
                     elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name+'/biases:0']))
                except:
                     if 'weights' in var.name:
                        init_ops.append(var.assign(d[w_name][0]))
                     elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name][1]))
    return init_ops


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(tf.sub(x, y))
    d = tf.sqrt(tf.reduce_sum(d))  # What about the axis ???
    return d


def square_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    with tf.name_scope("triplet_loss"):
        d_p_squared = square_distance(anchor_feature, positive_feature)
        d_n_squared = square_distance(anchor_feature, negative_feature)
        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)


def main(subset, sketch_dir, image_dir, sketch_dir_te, image_dir_te, triplet_path, mean, hard_ratio, batch_size,
         phase, phase_te, net_model):

    iterations = 1000
    validation_test = 200
    margin = 0.3
    save_step = 200
    model_path = "./model_temp/%s/%s_%d/" % (subset, net_model, iterations)
    pre_trained_model = './model/sketchnet_init.npy'
    # pre_trained_model = './model_def10/%s/%s_%d/model.npy' % (subset, net_model, iterations)
    if net_id == 1:
        model_path = "./model_r1/%s/%s_stage2_%d/" % (subset, net_model, iterations)
        pre_trained_model = './model_def10/%s/%s_stage2_%d/model.npy' % (subset, net_model, iterations)

    pre_step = 0
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Siamese place holders
    train_anchor_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="anchor")
    train_positive_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="positive")
    train_negative_data = tf.placeholder(tf.float32, shape=(None, 225, 225, 1), name="negative")

    # Creating the architecture
    if net_model == 'deep_sbir':
        with tf.variable_scope("sketch_a_net_sbir"):
            train_anchor = sketch_a_net_sbir(tf.cast(train_anchor_data, tf.float32) - mean, True)
            tf.get_variable_scope().reuse_variables()
            train_positive = sketch_a_net_sbir(tf.cast(train_positive_data, tf.float32) - mean, True)
            train_negative = sketch_a_net_sbir(tf.cast(train_negative_data, tf.float32) - mean, True)
    elif net_model == 'DSSA':
        with tf.variable_scope("sketch_a_net_sbir"):
            train_anchor = sketch_a_net_dssa(tf.cast(train_anchor_data, tf.float32) - mean, True)
            tf.get_variable_scope().reuse_variables()
            train_positive = sketch_a_net_dssa(tf.cast(train_positive_data, tf.float32) - mean, True)
            train_negative = sketch_a_net_dssa(tf.cast(train_negative_data, tf.float32) - mean, True)
    else:
        print('Please define the net_model')

    init_ops = init_variables(pre_trained_model)
    loss, positives, negatives = compute_triplet_loss(train_anchor, train_positive, train_negative, margin)

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = 0.001
    data_sampler = triplet_sampler_asy.TripletSamplingLayer()
    data_sampler_te = triplet_sampler_asy.TripletSamplingLayer()
    data_sampler.setup(sketch_dir, image_dir, triplet_path, mean, hard_ratio, batch_size, phase)
    data_sampler_te.setup(sketch_dir_te, image_dir_te, triplet_path, mean, hard_ratio, batch_size, phase_te)
    optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate).minimize(loss, global_step=batch)

    # validation_prediction = tf.nn.softmax(lenet_validation)
    # saver = tf.train.Saver(max_to_keep=5)
    dst_path = './log'
    model_id = '%s_%s_log.txt' % (subset, net_model)
   #  if net_id == 1:
    #     model_id = '%s_%s_stage2_log.txt' % (subset, net_model)

    filename = dst_path+'/'+model_id

    # Training
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        session.run(init_ops)
        for step in range(iterations):
            f = open(filename, 'a')
            batch_anchor, batch_positive, batch_negative = data_sampler.get_next_batch()

            feed_dict = {train_anchor_data: batch_anchor,
                         train_positive_data: batch_positive,
                         train_negative_data: batch_negative}
            _, l1 = session.run([optimizer, loss], feed_dict=feed_dict)
            # save_path = saver.save(session, model_path, global_step=step)
            print("Iter %d: Loss Train %f" % (step+pre_step, l1))
            f.write("Iter "+str(step+pre_step) + ": Loss Train: " + str(l1))
            f.write("\n")
            # train_writer.add_summary(summary, step)

            if step % save_step == 0:
                str_temp = '%smodel-iter%d.npy' % (model_path, step+pre_step)
                save_dict = {var.name: var.eval(session) for var in tf.global_variables()}
                np.save(str_temp, save_dict)

            if step % validation_test == 0:
                batch_anchor, batch_positive, batch_negative = data_sampler_te.get_next_batch()

                feed_dict = {train_anchor_data: batch_anchor,
                             train_positive_data: batch_positive,
                             train_negative_data: batch_negative
                             }

                lv = session.run([loss], feed_dict=feed_dict)
                # test_writer.add_summary(summary, step)
                print("Loss Validation {0}".format(lv))
            #     f.write("Loss Validation: " + str(lv))
            #     f.write("\n")
            # f.close()


if __name__ == '__main__':
    # 'deep_sbir'(the model of cvpr16) or 'DSSA'(the model of iccv17)
    def10 = 0

    net_model = 'DSSA'
    subset = 'handbags'
    mean = 250.42
    hard_ratio = 0.75
    batch_size = 128
    phase = 'TRAIN'
    phase_te = 'TEST'
    base_path = './data'
    sketch_dir = '%s/%s/%s_sketch_db_%s.mat' % (base_path, subset, subset, phase.lower())
    image_dir = '%s/%s/%s_edge_db_%s.mat' % (base_path, subset, subset, phase.lower())
    triplet_path = '%s/%s/%s_annotation.json' % (base_path, subset, subset)  # pseudo annotations for handbags
    sketch_dir_te = '%s/%s/%s_sketch_db_%s.mat' % (base_path, subset, subset, phase_te.lower())
    image_dir_te = '%s/%s/%s_edge_db_%s.mat' % (base_path, subset, subset, phase_te.lower())

    if def10 == 1:
        sketch_dir = '%s/%s/%s_sketch_db_%s_def10.mat' % (base_path, subset, subset, phase.lower())
        triplet_path = '%s/%s/%s_annotation_def10.json' % (base_path, subset, subset)

    main(subset, sketch_dir, image_dir, sketch_dir_te, image_dir_te, triplet_path, mean, hard_ratio, batch_size, phase, phase_te, net_model)
