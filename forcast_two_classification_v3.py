import os
import sys
import re
import numpy as np
from zhon.hanzi import punctuation
import jieba
import tensorflow as tf
from gensim.models import word2vec
from two_classification import TextCNN
import two_dealDate


tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

FLAGS = tf.flags.FLAGS

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
    with sess.as_default() as session:
        categorys=two_dealDate.get_all_categorys()
        cnn = TextCNN(
            sequence_length=two_dealDate.sentence_max_length,
            num_classes=two_dealDate.num_class,
            filter_sizes=two_dealDate.filter_sizers,
            num_filters=two_dealDate.num_filters,
            l2_reg_lambda=two_dealDate.l2_reg_lambda)
        saver = tf.train.Saver(tf.global_variables())
        out_dir = os.path.abspath(os.path.join(two_dealDate.runPath, "two_models"))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)

        w = open('/export/user/shizhengxin/ques2id_cnn/qq_dui_score0.1_0.8.txt','r',encoding='utf-8')
        ws = open('/export/user/shizhengxin/user_faq/fenji_cnn_faq/qq_dui_filter_0.1_0.8.txt','a',encoding = 'utf-8')
        line = w.readline()

        while line:
            # Get token-ids for the input sentence.
            sentence_line = line
            sentences_line=sentence_line.rstrip('\n').split('----')
            all_result = []
            times = 0
            for sentence in sentences_line:

                sentence = re.sub("[%s]+" % punctuation, "", sentence)
                sentence=jieba.cut(sentence)
                sentence=[word for word in sentence]
                sentences=[sentence]
            # print(sentence)
            # sentence=np.array(sentence)
            # print(sentence.shape)
                sentences=two_dealDate.word2index(word_index,sentences)
                sentences=two_dealDate.batch_word_padding(two_dealDate.sentence_max_length,sentences)
            # sentences=np.array(sentences)
            # sentences=np.reshape(sentences,[-1,dealData.sentence_max_length])

                feed_dict = {
                cnn.input_x: sentences,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                output = session.run([cnn.probability],feed_dict=feed_dict)
                result={}
            # output_=tf.reshape(output,shape=[-1,dealData.num_class])
            # topk=tf.nn.top_k(output_,3)
            # topk=sess.run(topk)
            # print(topk)
            # output=np.array(output)
            #
            # print("--------------")
            # print(len(output))
            # print(len(output[0]))
            # print(output.shape)
                for i in range(len(output[0][0])):
                    result[categorys[i]]=output[0][0][i]
                    # p=output[0][0][i]
                    # print(line+"--"+categorys[i]+":"+str(p))
                result_sorted=sorted(result, key=result.get, reverse=True)
                all_result.append(result_sorted[0])
                times += 1
                if times == 2:
                    break
            new_result = set(all_result)
            if len(new_result) == 1 and new_result.pop() != '其他问题':
                
                ws.writelines('----'.join(sentences_line)+'\n')

            # for word in result_sorted:
            #     print(word+":"+str(result[word]))

            line = w.readline()
        w.close()
        ws.close()
