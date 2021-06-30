#coding=utf8
#KNN.py
import numpy as np 
import operator  #导入运算符模块
 
def createDataSet():   #创造一个数据集
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])   #构造特征集矩阵
    labels = ['A','A','B','B']     #输入标签
    return group,labels
 
 
"""函数功能就是根据数据集，然后预测新给定的样本数据InX的标签是什么（会在最后返回）"""
def classify(inX,dataSet,labels,k):    #inX测试集，Dataset是已知数据集
 
	#第一步：求欧式距离
	dataSetSize=dataSet.shape[0]         #dataSetSize是dataSet的行数，用上面的举例就是4行
	diffMat= np.tile(inX,(dataSetSize,1))-dataSet   #坐标相减，tile意思是将inX（比如inx为[1,2,1]）构建成一个矩阵和已知的点数集合dataSet相减
	sqDiffMat=diffMat**2                 #上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方
	sqDistances=sqDiffMat.sum(axis=1)    #axis=1是列相加，，这样得到了(x1-x2)^2+(y1-y2)^2
	distances=sqDistances**0.5           #开根号，这个之后才是距离
 
	#第二步：所有距离进行排序
	sortedDistIndicies=distances.argsort()         #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
 
	#第三步：取排序前k个，这里的k就是KNN中的k，然后放入到词典中，词典的结构是：  类别：个数
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]  #设置词典中的key值，也就是标签
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1    #设置词典中的key值对应的value值
        #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的）
 
	#第四步：将字典排序，然后返回个数最多的类别
	soredClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
	return soredClassCount[0][0]             #返回类别最多的类别
 
group,labels = createDataSet()
predict = classify([0,0],group,labels,3)  #装入一个新的数据，然后开始测试类别
print(predict)
