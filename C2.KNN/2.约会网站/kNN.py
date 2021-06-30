import numpy as np  # 导入科学计算包
import operator  # 导入运算符模块

"""
函数1：读取数据集（一般是从文本中）
这个函数其实是createDataSet的改进版，createDataSet只是随机创建了一个非常简单的特征数据集和对应的标签
而这里的file2matrix函数，则是读取文本中的数据集，然后转换成Python可以识别的数据集，
其中returnMat相当于group，是特征数据集；而classLabelVector是labels是标签集   
"""


def file2matrix(filename):
    # 第一步：读取指定的文本，然后将文本一行行拆分成列表
    fr = open(filename)
    arrayOLines = fr.readlines()  # 将所有数据都读取出来，然后划分成列表，每个元素是一行
    numberOfLines = len(arrayOLines)  # 返回列表arrayOLines的元素个数，也就是文本行数
    returnMat = np.zeros((numberOfLines, 3))  # 创造一个numberOFLines行3列的矩阵，每个元素皆为0
    classLabelVector = []  # 分类标签向量
    index = 0

    # 第二步：针对每一行制成矩阵中的一行元素，和标签向量中的一个元素
    for line in arrayOLines:  # 依次对文本每一行进行操作
        line = line.strip()  # 删除每行文本中的回车符
        listFromLine = line.split('\t')  # 切割line字符串，以\t为分割点，这是制表符
        returnMat[index, :] = listFromLine[0:3]  # 截取listFormLine第一位到第三位的字符，然后赋值给numberOFLines行3列的矩阵，index代表行数
        classLabelVector.append(int(listFromLine[-1]))  # 将每一行最后一位（也就是标签），赋值给类标签向量，注意要强制转换
        index += 1

    # 第三步：最后返回特征数据集和分类标签
    return returnMat, classLabelVector


'''
函数2：归一化特征值，使所有特征值变化范围一致，这里是使得数据集都为0-1之间
'''


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 将每列中最小值赋给minVals，注意minVals是矩阵1*3（因为上面数据集只有3列）
    maxVals = dataSet.max(0)  # 同上面，只是这里取出的是最大值
    ranges = maxVals - minVals  # 这里是每列的最大波动范围（最大值-最小值），ranges的规模与minVals是一致的
    normDataSet = np.zeros(np.shape(dataSet))  # 创造一个0矩阵，规模和数据集一致
    m = dataSet.shape[0]  # 求数据集的行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 后面的tile函数是创建一个规模和dataSet一样的矩阵，每行都是minVals
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 两个规模相同的矩阵对应元素相除，使得每个元素范围在0-1之间
    return normDataSet, ranges, minVals  # 返回归一后的数据集，波动范围，还有最小值，可以根据这三个值对数据集进行还原


"""
函数3：分类器（核心）
函数功能就是根据数据集，然后预测新给定的样本数据InX的标签是什么（会在最后返回）
"""


def classify0(inX, dataSet, labels, k):
    # 第一步：求欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 第二步：所有距离进行排序
    sortedDistIndicies = distances.argsort()

    # 第三步：取排序前k个，这里的k就是KNN中的k，然后放入到词典中，词典的结构是：类别：个数
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 第四步：将字典排序，然后返回个数最多的类别
    soredClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return soredClassCount[0][0]


'''
函数4：测试函数
这里其实不算KNN核心之中了，因为它对于算法没有任何提升，只是告诉你：你的算法准确率多少
这个算是辅助函数吧，让你看清你的算法效率怎么样，准确率低，就改善算法
如果准确率高，那么恭喜你，你的算法可以用于商业了
'''


def datingClassTest():
    # 第一步：准备测试数据集
    hoRatio = 0.10  # 海伦给出的数据集要有一部分用于测试集，这是比例
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')  # 读取文本数据集，然后返回可以用数据集和标签
    normMat, ranges, minVals = nutoNorm(datingDataMat)  # 数据处理：自动归一化
    m = normMat.shape[0]  # 数据集的行数，也可以说是样本数，或者曾经约会过的人数
    numTestVecs = int(m * hoRatio)  # 测试集的数量：比例*总数量

    errorCount = 0.0  # 这个是错误数量，机器学习算法都要有一个正确率或者错误率

    # 第二步：开始测试这个算法了，说白了还是测试classify0函数，一共测试了numTeestVecs
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],
                                     3)  # 调用分类器，返回分类结果
        print('分类器预测的结果是: %d,真实结果是: %d' % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0  # 每当有一个预测结果与真实结果（也称标签）不符，错误数量就加1，
    print('总的错误率是: %f' % (errorCount / float(numTestVecs)))  # 错误数量除以总数量，就是错误率了


"""
函数5：预测新样本
"""


def classifyPerson():
    # 第一步：指定结果标签
    resultList = ['不喜欢', '一般喜欢', '特别喜欢']  # 结果标签

    # 第二步：通过你的输入得到新样本特征
    percentTats = float(input("玩游戏百分比?"))  # 读取你的输入数据
    ffMiles = float(input('每年飞行里程数?'))  # 问个问题，你输入数据，它读入
    iceCream = float(input('每周消耗冰淇淋公升数?'))  # 输入数据赋值，并类型转换

    # 第三步：得到已知标签的数据集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读取数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据自动归一化
    inArr = np.array([ffMiles, percentTats, iceCream])  # 将读入的数据转换成一个向量

    # 第四步：将新样本和已知标签的样本数据集输入分类器，得到预测结果
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)  # 调用分类器
    print("你对这个人的喜欢程度可能是：", resultList[classifierResult - 1])  # 输出结果


classifyPerson()  
