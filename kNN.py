
# coding: utf-8

# In[1]:


import numpy as np
import operator


# In[ ]:


#创建数据集和标签
def createdataset():
    group=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group, labels


#K-近邻算法
def classify0(inX,dataset,labels,k):  
    '''
    inX是用于分类的输入向量，输入的训练样本集为dataset,标签向量为labels,k表示用于选择最近邻居的数目
    '''
    datasetsize=dataset.shape[0]   #假设dataset是一个n*m矩阵，shape[0]返回n的值#
    '''
    计算已知类别数据集中的点与输入点之间的距离
    '''
    diffmat=np.tile(inX,(datasetsize,1))-dataset
    sqdiffmat=diffmat**2
    sqdistances=sqdiffmat.sum(axis=1)
    distances=(sqdistances)**0.5
    '''
    按照距离递增次序排序
    '''
    sorteddistindicies=distances.argsort()  #argsort返回排序后的index
    '''
    选择距离最小的k个点
    '''
    classcount={}
    for i in range(k):
        voteIlabel=labels[sorteddistindicies[i]]
        classcount[voteIlabel]=classcount.get(voteIlabel,0)+1   #get(voteIlabel,0) 返回label对应的值，没有则对应0
    '''
    排序
    '''
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)  
    #items() 返回classcount的list      
    #itemgetter(1)按照第二个元素的次序对元组进行排序
    #按照key逆序排列（从大到小排列）
    return sortedclasscount[0][0]  #返回频率最高的类别


'''
示例：使用K-近邻算法改进约会网站的配对效果
'''
# 讲文本记录转换为NumPy的解析程序
#输入字符串，输出为训练样本矩阵和类标签向量
def file2matrix(filename):
    #导入txt文件
    fr=open(filename)
    arrayolines=fr.readlines()
    #得到文件行数
    numberoflines=len(arrayolines)
    '''
    创建返回的NumPy矩阵
    以零填充的矩阵
    '''
    returnmat=np.zeros((numberoflines,3))
    classlabelvector=[]
    
    '''
    解析文件数据到列表
    '''
    index=0
    for line in arrayolines:
        line=line.strip()                         #strip()删除分行符
        listfromline=line.split('\t')             #根据tab(\t)分割成一个元素列表
        returnmat[index,:]=listfromline[0:3]
        classlabelvector.append(int(listfromline[-1]))   #[-1]指最后一列数据
        index+=1
    return returnmat,classlabelvector


# 归一化特征值
## 特征值取值范围[0,1]
## newvalue=(oldvalue-min)/(max-min)

def autonorm(dataset):
	minvals=dataset.min(0)           #min(0)选取列的最小值，min()选取行的最小值
	maxvals=dataset.max(0)
	ranges=maxvals-minvals
	normdataset=zeros(np.shape(dataset))
	m=dataset.shape[0]
	normdataset=dataset-np.tile(minvals,(m,1))          #tile函数将变量复制成输入矩阵同样大小的矩阵，便于矩阵计算
	normdataset=normdataset/np.tile(ranges,(m,1))
	return normdataset, ranges,minvals


#测试算法：作为完整程序验证分类器

def datingclasstest():
    horatio=0.1                  # 可通过改变training dataset 比例来检验错误率是否随着变量值的变化而增加
    m=normmat.shape[0]
    numtestvecs=int(m*horatio)
    errorcount=0
    for i in range(numtestvecs):
        classifierresult=classify0(normmat[i,:],normmat[numtestvecs:m,:],datinglabels[numtestvecs:m],3)
        print('the classifier came back with: %d, the real answer is:%d'%(classifierresult,datinglabels[i]))
        if(classifierresult !=datinglabels[i]):errorcount+=1
    print ('the total error rate is: %f'%(errorcount/float(numtestvecs)))


#使用算法：构建完整可用系统
## 约会网站预测函数
def classifyperson():
    resultlist=['not at all', 'in small doses','in large doses']
    # python3里raw_input变成了input，用户手动输入
    percenttats=float(input('percentage of time spent playing video games?'))
    ffmiles=float(input('frequent flier miles earned per year?'))
    icecream=float(input('liters of ice cream consumed per year?'))
    inarr=np.array([ffmiles,percenttats,icecream])
    classifierresult=classify0((inarr-minvals)/ranges,normmat,datinglabels,3)    #输入值带入公式
    print('You will probably like this person: ',resultlist[classifierresult-1])




'''
示例：手写识别系统
识别数字0到9
'''

# 准备数据：将图像转换为测试向量
## 将32*32的二进制图像矩阵转换为1*1024的向量

def img2vector(filename):
    returnvect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr=fr.readline()      #readline()返回每一行的值
        for j in range(32):
            returnvect[0,32*i+j]=int(linestr[j])
    return returnvect

# 测试算法：使用k-近邻算法识别手写数字
from os import listdir

def handwritingclasstest():
    hwlabels=[]
    trainingfilelist=listdir('trainingDigits')       #listdir(path)输出指定路径下所有文件名
    m=len(trainingfilelist)
    trainingmat=zeros((m,1024))
    '''
    从文件名解析分类数字
    '''
    for i in range(m):
        filenamestr=trainingfilelist[i]
        filestr=filenamestr.split('.')[0]
        classnumstr=int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i,:]=img2vector('trainingDigits/%s'%filenamestr)
    testfilelist=listdir('testDigits')
    errorcount=0
    mtest=len(testfilelist)
    for i in range(mtest):
        filenamestr=testfilelist[i]
        filestr=filenamestr.split('.')[0]
        classnumstr=int(filestr.split('_')[0])
        vectorundertest=img2vector('testDigits/%s'% filenamestr)
        classifierresult=classify0(vectorundertest,trainingmat,hwlabels,3)
        print('the classifier came back with:%d, the real answer is:%d'%(classifierresult,classnumstr))
        if(classifierresult !=classnumstr):errorcount+=1
        print ('\nthe total number of errors is:%d'% errorcount)
        print('\nthe total error rate is:%f'%(errorcount/float(mtest)))

