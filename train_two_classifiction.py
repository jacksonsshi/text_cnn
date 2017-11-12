#! /usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import os
import time
import datetime
import two_dealDate
from two_classification import TextCNN
from tensorflow.contrib import learn
from keras.preprocessing import sequence
from gensim.models import word2vec

# Training
# ==================================================
max_len = 30
w2v=word2vec.Word2Vec.load(two_dealDate.word2vecPath)
# vec=w2v.wv['京东']
vocab=sorted([word for word in w2v.wv.vocab])

word_index=dict([(y, x) for (x, y) in enumerate(vocab)])

embedding_matrix = np.zeros((len(vocab) + 1, 300))
for i,word in enumerate(vocab):
    embedding_vector = w2v.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i+1] = embedding_vector



with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=two_dealDate.sentence_max_length,
            num_classes=two_dealDate.num_class,
            filter_sizes=two_dealDate.filter_sizers,
            num_filters=two_dealDate.num_filters,
            l2_reg_lambda=two_dealDate.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        out_dir = os.path.abspath(os.path.join(two_dealDate.runPath, "two_models"))
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])

        # Dev summaries

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=out_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("create fresh net")
            sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)
            print("step {}, loss {:g}".format( step, loss))


        # Generate batches
        steps = two_dealDate.max_step
        # Training loop. For each batch...
        for step in range(steps):
            #x_text, y_text = dealData.category_class_batch_data()
            x_text, y_text = two_dealDate.get_category_class_batch_data()

            x_text = two_dealDate.word2index(word_index, x_text)
            # x_text = sequence.pad_sequences(x_text, maxlen=max_len)
            # x_text=tf.pad()
            x_text=two_dealDate.batch_word_padding(two_dealDate.sentence_max_length,x_text)
            train_step(x_text, y_text)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_dir, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
