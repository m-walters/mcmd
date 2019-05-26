
# coding: utf-8

# In[ ]:


def read_lbl(filename_queue, batchSize):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE4_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    
    image_shape = tf.stack([height, width, 4])
    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant((constHeight, constWidth, 4), dtype=tf.int32)
    
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=constHeight,
                                           target_width=constWidth)
    blankvec = tf.Variable([0,0,0,0,0,0])    
    
    return resized_image, label, blankvec

def LblBatch(filename_queue, batchSize):
    prepped_example = read_lbl(filename_queue, batchSize)
# trnbatch = tf.train.shuffle_batch_join(trnQlist,
#                                         batch_size=BATCHSIZE,
#                                         capacity=1000+3*BATCHSIZE,
#                                         min_after_dequeue=1000)
    imagebatch, labelbatch, blankvecs = tf.train.shuffle_batch_join(
                                          [prepped_example[0], prepped_example[1], blankvec],
                                          batch_size=batchSize,
                                          capacity=1000 + batchSize*3,
                                          min_after_dequeue=1000)
    
    return imagebatch, labelbatch, blankvecs


# In[ ]:


def read_unlbl(filename_queue, batchSize):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 4])
    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant((constHeight, constWidth, 4), dtype=tf.int32)
    
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=constHeight,
                                           target_width=constWidth)
    
    return resized_image


def UnlblBatch(filename_queue, batchSize):
    prepped_example = read_unlbl(filename_queue, batchSize)
    return tf.train.shuffle_batch([prepped_example],
                                          batch_size=batchSize,
                                          capacity=batchSize*2,
                                          num_threads=2,
                                          min_after_dequeue=0)


# In[ ]:


STRIDE = 4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    s = STRIDE
    return tf.nn.conv2d(x, W, strides=[1,s,s,1], padding='SAME')

def assign_labels():
    return


x = tf.placeholder(tf.float32, shape=[None, constHeight, constWidth, 4])
y_ = tf.placeholder(tf.float32, shape=[None, 6])

x_image = tf.reshape(x, [-1, constHeight, constWidth, 4])

W_conv1 = weight_variable([5, 5, 4, 32]) # [x,y,nInputChannel,nOutChannel]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# Dividing by STRIDE*STRIDE for each conv layer
len_fc1 = constHeight*constWidth*64/(STRIDE*STRIDE*STRIDE*STRIDE)
W_fc1 = weight_variable([len_fc1, 512])
b_fc1 = bias_variable([512])
h_conv2_flat = tf.reshape(h_conv2, [-1, len_fc1])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 6])
b_fc2 = bias_variable([6])

y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
trn_queue = tf.train.string_input_producer(trnrecords)
trnQlist = [read_lbl(trn_queue, BATCHSIZE) for _ in range(len(trnrecords))]
test_queue = tf.train.string_input_producer([testrecords[0]], num_epochs=1)
unlbl_queue = tf.train.string_input_producer([unlblrecords[0]], num_epochs=1)

# Even when reading in multiple threads, share the filename
# queue.
BATCHSIZE = 30
# trnbatch = LblBatch(trn_queue, BATCHSIZE)
trnbatch = LblBatch(trnQlist, BATCHSIZE)
# trnbatch = tf.train.shuffle_batch_join(trnQlist,
#                                         batch_size=BATCHSIZE,
#                                         capacity=1000+3*BATCHSIZE,
#                                         min_after_dequeue=1000)
# testbatch = LblBatch(test_queue, BATCHSIZE)
# unlblbatch = UnlblBatch(unlbl_queue, BATCHSIZE)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

print "About to start sess"

with tf.Session() as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    print "Initialized sess"
    
    for i in xrange(1,4):
    
        trnbatch_py = sess.run([trnbatch[0], trnbatch[1], trnbatch[2]])
        testbatch_py = sess.run([testbatch[0], testbatch[1], testbatch[2]])
        
        print "step", i
        # Assign proper labels to the lblvecs
        # Currently lblvecs is a batch of blank (zeros) rows
        for l in range(len(trnbatch_py[1])):
            trnbatch_py[2][l][trnbatch_py[1][l]] = 1
        print trnbatch_py[2]

#         for p in trnbatch_py[0][0:6]:
#             io.imshow(p)
#             io.show()

        if i % 2 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: trnbatch_py[0], y_: trnbatch_py[2], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: trnbatch_py[0], y_: trnbatch_py[2], keep_prob: 0.5})

#         print('test accuracy %g' % accuracy.eval(feed_dict={
#             x: images, y_: lblvecs, keep_prob: 1.0}))
        
    print "Closing sess"
    coord.request_stop()
    coord.join(threads)

