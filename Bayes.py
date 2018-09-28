#-*-coding:utf-8-*-
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy
import re
import os
import random



def load_data(folder_path):
    print("Loading dataset ...")
    loadTime = time()
    datalist = datasets.load_files(folder_path)
    #datalist是一个Bunch类，其中重要的数据项有
    #data:原始数据
    #filenames:每个文件的名称
    #target:类别标签（子目录的文件从0开始标记了索引）
    #target_names:类别标签（子目录的具体名称）
    #输出总文档数和类别数
    print("summary: {0} documents in {1} categories.".format(len(datalist.data),len(datalist.target_names)))
    #加载数据所用的时间
    print("Load data in {0}seconds".format(time() - loadTime))
    #去停用词操作
    #datalist.data = [word for word in datalist.data if(word not in stopwords.words('english'))]
    return datalist

def vector_represent(ori_data):
    print("Vectorizing dataset ...")
    vectorTime = time()
    #TfidfVectorizer将数据文档转换为特征矩阵
    #stop_words直接使用英文的停用词表进行去停用词操作
    #latin-1编码是8比特的字符集，定义了256个字符
    vectorizer = TfidfVectorizer(stop_words="english", encoding="latin-1")
    #fit()进行语料库分析，创建词典等操作
    #transform把每个email转化为TF-IDF向量
    x_train = vectorizer.fit_transform(data for data in ori_data.data)
    #文档数目和特征词语数
    print("n_samples:%d,n_features:%d" % x_train.shape)
    #举例说明第0个文档的文件中的特征数
    print("number of non-zero features in sample[{0}]:{1}".format(ori_data.filenames[0],x_train[0].getnnz()))
    print("Vector data in {0}seconds".format(time() - vectorTime))
    return x_train,vectorizer

def train_model(x_train, y_train):
    print("Train model ...")
    TrainTime = time()
    # 使用朴素贝叶斯多项式模型
    #clf = MultinomialNB(alpha=0.001)
    x_train = x_train.todense()
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    print("train score: {0}".format(train_score))
    print("Train data in {0}seconds".format(time()-TrainTime))
    return clf

def test_model(clf,testdata_path,vectorizer):
    test_list = load_data(testdata_path)
    X_test = vectorizer.transform(data for data in test_list.data)
    Y_test = test_list.target
    print("n_samples:%d,n_features:%d" % X_test.shape)
    print("number of non-zero features in sample[{0}]:{1}".format(test_list.filenames[0], X_test[0].getnnz()))
    #print("Vector data in {0}seconds".format(time() - vectorTime))
    pred = clf.predict(X_test[0])
    print("predict: {0} is in category {1}".format(
        test_list.filenames[0], test_list.target_names[pred[0]]))
    print("actually: {0} is in category {1}".format(
        test_list.filenames[0], test_list.target_names[test_list.target[0]]))
    print("Predicting test dataset ...")
    PredictTime = time()
    pred = clf.predict(X_test)
    print("Predict in {0}seconds".format(time()-PredictTime))
    print("Classification report on test set for classifier:")
    print(clf)
    print(classification_report(Y_test, pred, target_names=test_list.target_names))
    #生成混淆矩阵
    con_matri = confusion_matrix(Y_test, pred)
    print("Confusion matrix:")
    print(con_matri)
    return con_matri

def draw_confusionMatrix(confusion):
    plt.figure(figsize=(3, 3), dpi=144)
    plt.title('Confusion Matrix')
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.matshow(confusion, fignum=1, cmap='gray')
    plt.colorbar()
    plt.show()

# if __name__ == "__main__":
#     #训练集和测试集的路径
#     train_path = "/home/stardream/DDML/MachineLearning/spamClassification/spamDataset/email/train"
#     test_path = "/home/stardream/DDML/MachineLearning/spamClassification/spamDataset/email/test"
#     #导入训练集并进行预处理操作
#     train_list = load_data(train_path)
#     #将文档表示为向量，并进行TF-IDF处理
#     X_train,vector = vector_represent(train_list)
#     #获得训练集的标签
#     Y_train = train_list.target
#     #训练模型
#     classification = train_model(X_train, Y_train)
#     #测试模型
#     confusion = test_model(classification,test_path,vector)
#     draw_confusionMatrix()


def word_create(ori_data):
    #还没有进行去停用词的操作nltk.stopwoords的包没能下载下来
#    ori_data.data = [word for word in ori_data.data if(word not in stopwords.words('english'))]
    print("Vectorzing dataset ...")
    #建立一个集合列表
    word_dic = set([])
    #词向量的时间
    vectorTime = time()
    #词典的构造
    for doc in ori_data.data:
        #doc是byte，这里将byte转化为string
        doc = str(doc, encoding = "utf-8")
        #使用正则表达式将特殊符号去除
        doc = re.sub("[\s+\.\!\/_,$%^*(+\"\'-]+|[+——！，。？、~@#￥%……&*（）<>]+", " ", doc)
        #使用默认的空格方式将email分隔开，然后转化为小写字母，与原集合取并集
        word_dic = word_dic|set(doc.lower().split())
    #向量化的时间和词典中词的数量
    print("Vectorzing time:{0}\nThe number of word_dictionary:{1}".format(vectorTime,len(word_dic)))
    return list(word_dic)

def doc_represent(wordDic,ori_data):
    #创建一个文档数（行）*词向量（列）长度的二维数组
    doc_re = numpy.zeros((len(ori_data.data),len(wordDic)),dtype= numpy.int)
    #计数器
    count = 0
    #用来记录词向量表示时间
    representTime = time()
    for doc in ori_data.data:
        #同word_create函数，进行同样的操作
        doc = str(doc, encoding = "utf-8")
        doc = re.sub("[\s+\.\!\/_,$%^*(+\"\'-]+|[+——！，。？、~@#￥%……&*（）<>]+", " ", doc)
        for word in doc.lower().split():
            if word in wordDic:
                #将对应词向量位置置1
                doc_re[count][wordDic.index(word)] = 1
        count = count+1
    print("Represent doc time:{0}\nThe number of doc:{1}".format(representTime-time(),len(doc_re)))
    #返回表示文档的二维数组
    return doc_re

def pre_probabilty(ori_data):
    s_pre_pro = []

    #正常邮件的先验概率
    P_normal = (normal + 1.0)/(len(ori_data.data) + 2.0)
    s_pre_pro.append(P_normal)
    #垃圾邮件的先验概率
    P_spam = (spam + 1.0)/(len(ori_data.data) + 2.0)
    s_pre_pro.append(P_spam)
    #返回先验概率的列表
    return s_pre_pro

#计算每个词在正常邮件垃圾邮件中的数目
def wordNum_email(email_repre,wordDic):
    #用二维向量存储
    num_word = numpy.zeros((2,len(wordDic)),dtype= numpy.int)
    for i in range(len(wordDic)):
        #在正常邮件的数目
        for j in range(normal):
            num_word[0][i] += email_repre[j][i]
        #在垃圾邮件中的数目
        for j in range(normal, spam+normal):
            num_word[1][i] += email_repre[j][i]
    return num_word

#条件概率
def con_probabilty(email_repre,wordDic):
    #得到每个词汇在正常邮件、垃圾邮件中的数目
    word_num = wordNum_email(email_repre,wordDic)
    word_pro = numpy.zeros((2,len(wordDic)),dtype = numpy.double)
    for i in range(len(wordDic)):
        word_pro[0][i] = round((word_num[0][i]+1)/(normal + 2),8)
        word_pro[1][i] = round((word_num[1][i]+1)/(spam + 2 ),8)
    return word_pro

#得到每个类别中的文档数
def class_num(path,class_name):
    count = 0
    path=path+"/"+class_name
    for root, dirs, files in os.walk(path):  # 遍历统计
        for each in files:
            count += 1
    return count

#测试
def test_spam(test_repre,pre_pro,con_pro):
    email_pro = numpy.zeros((len(test_repre),2),dtype = numpy.double)
    email_judge = []
    normal_num = 0
    spam_num = 0
    for i in range(len(test_repre)):
        email_pro[i][0] = round(pre_pro[0],8)
        email_pro[i][1] = round(pre_pro[1],8)
        for j in range(len(test_repre[0])):
            if test_repre[i][j] != 0:
                email_pro[i][0] *= con_pro[0][j]
                email_pro[i][1] *= con_pro[1][j]
        if email_pro[i][0] > email_pro[i][1] :
            email_judge.append(0)
        elif email_pro[i][0] < email_pro[i][1] :
            email_judge.append(1)
        else :
            if random.random() > 0.5:
                email_judge.append(1)
            else:
                email_judge.append(0)
    for i in range(normal_test):
        if email_judge[i] == 0:
            normal_num +=1
    for i in range(normal_test,len(test_repre)):
        if email_judge[i] == 1:
            spam_num +=1
    print("email_judge\n")
    print(email_judge)
    print("normal_num="+str(normal_num)+"\nspam_num="+str(spam_num))
    return (normal_num + spam_num)/len(test_repre)

if __name__ == "__main__":
     # 训练集和测试集的路径
     train_path = "/home/stardream/DDML/MachineLearning/spamClassification/spamDataset/email/train1"
     test_path = "/home/stardream/DDML/MachineLearning/spamClassification/spamDataset/email/test1"
     train_list = load_data(train_path)
     test_list = load_data(test_path)
     # 正常邮件的数目
     normal = class_num(train_path,"pos")
     # 垃圾邮件的数目
     spam = class_num(train_path,"neg")
     #建立词汇表
     WordDictionary = word_create(train_list)
     #将训练数据进行向量表示
     docRepre = doc_represent(WordDictionary,train_list)
     #计算先验概率
     prePro = pre_probabilty(train_list)
     #计算条件概率
     conPro = con_probabilty(docRepre,WordDictionary)
     print("preProbablity:",prePro)
     print("conProbablity:",conPro)
     #测试数据的向量表示
     testRepre = doc_represent(WordDictionary,test_list)
     # 正常邮件的数目
     normal_test = class_num(test_path, "pos")
     # 垃圾邮件的数目
     spam_test = class_num(test_path, "neg")
     #测试数据的准确率
     test_accuracy = test_spam(testRepre,prePro,conPro)
     print ("test accuracy")
     print(test_accuracy)

