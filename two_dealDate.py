import re
import random
import numpy as np
import jieba
from zhon.hanzi import punctuation

runPath = "./"
word2vecPath = "/export/user/shizhengxin/word2vec/word2vec_test_v7.model"
question_category_classPath="/export/user/shizhengxin/field_cnn/all_ques_data.txt"
category_class_Path="/export/user/shizhengxin/field_cnn/ques_id.txt"
other_question_classPath = '/export/liaobin/webServer_all_faqs_2/data/key_model_delete/neg_example_all_v3.txt'

num_s=10
num_r=10
num_d=5
num_sample=100
num_category=17
sentence_max_length=30
num_class=17
embedding_size=300
filter_sizers=[3,4,5]
num_filters=128
l2_reg_lambda=0.1
max_step=20000

def word_padding(sentence_length,sentence):
    current_sentence_length = len(sentence)
    if current_sentence_length>sentence_length:
        return sentence[0:sentence_length]
    padded = [0 for _ in range(sentence_max_length - current_sentence_length)]
    sentence.extend(padded)
    return sentence
def batch_word_padding(sentence_length,sentences):
    result=[]
    for i in range(len(sentences)):
        sentence=word_padding(sentence_length,sentences[i])
        result.append(sentence)
    return result


def word2index(word_index,sentence_tokens):
    tokens=[]
    for sentence in sentence_tokens:
        token=[]
        for word in sentence:
            if word in word_index:
                token.append(word_index[word]+1)
            else:
                token.append(0)

        tokens.append(token)
    return tokens

def random_delete(sentence):
    sentence_length=len(sentence)
    result=[]
    max_delete_num=np.floor(sentence_length*0.3)
    if max_delete_num>0:
        delete_num=random.randint(1,max_delete_num)
        delete_words=[]
        for i in range(delete_num):
            r=random.randint(0,sentence_length-1)
            delete_words.append(r)
        for (index,word) in enumerate(sentence):
            if index not in delete_words:
                result.append(word)
        return result
    else:
        return sentence


def get_other_question_class_batch_data():
    questions = []
    labels = []
    otherfile=open(other_question_classPath,'r',encoding='UTF-8')
    lines=otherfile.readlines()
    random.shuffle(lines)
    i=0
    for line in lines:
        line=line.rstrip('\n')
        if i<1000:
            question = line
            question = re.sub("[%s]+" % punctuation, "", question)
            words = jieba.cut(question)
            words = [word for word in words]

            words = np.array(words)
            answer_index = num_class-1
            label = [0 for _ in range(num_class)]
            label[int(answer_index)] = 1
            questions.append(words)

            labels.append(label)

            shuffle = np.random.permutation(np.arange(len(words)))
            random_words = words[shuffle]
            questions.append(random_words)

            labels.append(label)

            deleted_words = random_delete(words)
            questions.append(deleted_words)

            labels.append(label)
            i+=1
        else:
            break
    return questions,labels

def get_category_class_batch_data():
    questions=[]
    labels=[]
    question_category_classfile=open(question_category_classPath,'r',encoding='UTF-8')
    lines=question_category_classfile.readlines();
    lines=[line.rstrip('\n') for line in lines]
    # print(len(lines))
    for i,line in enumerate(lines):

        line_tokens=line.replace('****','----').split("----")
        for sentence in line_tokens[1:]:
            
            
            question = sentence
            question=re.sub("[%s]+" %punctuation, "", line)
            words=jieba.cut(question)
            words=[word for word in words]
            words = np.array(words)
            categorys=line_tokens[0]
            categorys=categorys.split(',')
            label=[0 for _ in range(num_category)]
            for index in categorys:
                label[int(index)]=1
            questions.append(words)
            labels.append(label)

            shuffle = np.random.permutation(np.arange(len(words)))
            random_words = words[shuffle]
            questions.append(random_words)
            labels.append(label)


            deleted_words=random_delete(words)
            questions.append(deleted_words)
            labels.append(label)
    other_qs,  other_ls = get_other_question_class_batch_data()
    questions.extend(other_qs)
    labels.extend(other_ls)
    # print("get data finish")
    labels=np.array(labels)
    # print(len(labels))
    return questions,labels;

def category_class_batch_data():
    questions=[]
    labels=[]
    question_category_classfile=open(question_category_classPath,'r',encoding='UTF-8')
    lines=question_category_classfile.readlines();
    lines=[line.rstrip('\n') for line in lines]
    # print(len(lines))
    for i,line in enumerate(lines):
        line_tokens=line.replace('****','----').split("----")
        question=line_tokens[2]
        question=re.sub("[%s]+" %punctuation, "", line)
        words=jieba.cut(question)
        words=[word for word in words]
        words = np.array(words)
        categorys=line_tokens[0]
        categorys=categorys.split(',')
        label=[0 for _ in range(num_category)]
        for index in categorys:
            label[int(index)]=1
        questions.append(words)
        labels.append(label)
    print(len(labels))
    return questions,labels;


def get_all_categorys():
    category_classfile=open(category_class_Path,'r',encoding='UTF-8')
    categories=category_classfile.readlines()
    categories=[category.rstrip('\n').split('----')[1] for category in categories]
    return categories
