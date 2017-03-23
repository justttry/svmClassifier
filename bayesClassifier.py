#encoding:UTF-8

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import jieba
from numpy import *
import unittest

#----------------------------------------------------------------------
def loadfiles(dirs):
    """
    加载数据
    Parameter:
    Return:
    """
    return load_files(dirs)

#----------------------------------------------------------------------
def testClassifier():
    """"""
    #加载数据
    dataset = load_files('./test_file2')
    #对数据进行分词处理
    datasets = []
    for i in dataset.data:
        datasets.append(' '.join([j for j in jieba.cut(i)]))
    #生成训练数据集合测试集
    train_X, test_X, train_y, test_y = train_test_split(array(datasets),
                                                        dataset.target,
                                                        test_size=0.3)
    #训练数据
    classifier = bayesClassifier(MultinomialNB)
    classifier.trainFuncs(train_X, train_y)
    #预测数据
    counts = classifier.vectorizer.transform(test_X)
    tfidfs = classifier.transformer.transform(counts)
    print 'the precious is ', classifier.classifier.score(tfidfs, test_y)
    

#----------------------------------------------------------------------
def compareTextAB(A, B):
    """
    比较两个文本的相似度
    Parameter:
    A:文本A,格式为字符串
    B:文本B,格式为字符串
    Return:
    similar:相似度
    """
    #计算分词
    wordsA = jieba.cut(A)
    wordsA = [''.join([m, m]) if m >= u'\u4e00' and m <= u'\u9fa5' and len(m) == 1 else m for m in wordsA]
    wordsB = jieba.cut(B)
    wordsB = [''.join([m, m]) if m >= u'\u4e00' and m <= u'\u9fa5' and len(m) == 1 else m for m in wordsB]
    corpus = [' '.join(wordsA), ' '.join(wordsB)]
    #计算词频
    vectorizer = CountVectorizer(min_df=1, analyzer='word')
    X = vectorizer.fit_transform(corpus)
    #计算TFIDF
    transformer = TfidfTransformer(smooth_idf=False)
    X_tfidf = transformer.fit_transform(X)
    #计算余弦相似度
    similar = cosine_similarity(X)
    return similar[0, 1]


########################################################################
class bayesClassifier(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, classifier):
        """Constructor"""
        #基本分类器
        self.classifier = classifier()
        #词频生成函数
        self.vectorizer = CountVectorizer(min_df=1)
        #TFIDF生成函数
        self.transformer = TfidfTransformer(smooth_idf=False)
        
    #----------------------------------------------------------------------
    def predict(self, X):
        """
        预测类别
        Parameter:
        X:被预测矩阵
        Parameter:
        y:预测类
        """
        counts = self.vectorizer.transform(X)
        tfidfs = self.transformer.transform(counts)
        return self.classifier.transform(tfidfs)
    
    #----------------------------------------------------------------------
    def trainFuncs(self, X, y):
        """
        训练预处理机
        X:训练矩阵
        y:训练矩阵类别
        Return:
        None
        """
        #计算词频,训练词频函数
        counts = self.vectorizer.fit_transform(X)
        #计算tfidf，训练TFIDF函数
        tfidfs = self.transformer.fit_transform(counts, y)
        #训练分类器
        self.classifier.fit(tfidfs, y)


########################################################################
class SvmClassifierTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_loadfiles(self):
        dirs = './test_file2'
        datasets = loadfiles(dirs)
        print 'test_loadfiles done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_bayesClassifier_GaussianNB(self):
        """"""
        from sklearn import datasets
        iris = datasets.load_iris()
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
        print("Number of mislabeled points out of a total %d points : %d"
              % (iris.data.shape[0],(iris.target != y_pred).sum()))   
        print 'test_bayesClassifier Done'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_chi2(self):
        """"""
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        iris = load_iris()
        X, y = iris.data, iris.target
        m, n = X.shape
        t = int(m * 0.7)
        train_x = X[:t]
        train_y = y[:t]
        test_x = X[t:]
        test_y = y[t:]
        selects = SelectKBest(chi2, k=2)
        train_x_new = selects.fit_transform(train_x, train_y)
        test_x_new = selects.transform(test_x)
        print 'test_chi2 done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_tfidf(self):
        """"""
        from sklearn.feature_extraction.text import TfidfTransformer
        transformer = TfidfTransformer(smooth_idf=False)  
        train_counts = [[3, 0, 1],
                        [2, 0, 0],
                        [3, 0, 0],
                        [4, 0, 0],
                        [3, 2, 0],
                        [3, 0, 2]]
        test_counts = [[1, 2, 3], 
                       [0, 0, 3],
                       [1, 2, 1]]
        train_tfidf = transformer.fit_transform(train_counts)
        test_tfidf = transformer.transform(test_counts)
        print 'test_tfidf done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_CountVectorizer(self):
        """"""
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(min_df=1)
        corpus = ["我 来到 北京 清华大学",  
                  "他 来到 了 网易 杭研 大厦",  
                  "小明 硕士 毕业 与 中国 科学院 小明 ab",  
                  "我 爱 北京 天安门"] 
        X = vectorizer.fit_transform(corpus)        
        for i in vectorizer.get_feature_names():
            print i
        print X.toarray()
        print 'test_CountVectorizer done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_jieba(self):
        """"""
        import jieba
        corpus = ["我来到北京清华大学",  
                  "他来到了网易杭研大厦",  
                  "小明硕士毕业与中国科学院小明",  
                  "我爱北京天安门"]
        for i in corpus:
            print ' '.join(jieba.cut(i))
        print 'test_jieba done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_compareTextAB(self):
        """"""
        A = "我来到北京清华大学"
        B = "我爱北京天安门"
        print compareTextAB(A, B)
        A = "我爱中国"
        B = "我爱北京天安门"
        print compareTextAB(A, B)
        A = "我爱北京"
        B = "我爱北京天安门"
        print compareTextAB(A, B)
        print 'test_compareTextAB done'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_testClassifier(self):
        """"""
        testClassifier()
        
    #----------------------------------------------------------------------
    def test_pipline_cross_val_score(self):
        """"""
        #加载数据
        dataset = load_files('./test_file2')
        #对数据进行分词处理
        datasets = []
        for i in dataset.data:
            datasets.append(' '.join([j for j in jieba.cut(i)]))        #训练数据
        classifier = bayesClassifier(MultinomialNB)
        #预测数据
        classify = classifier.classifier
        vectorizer = classifier.vectorizer
        transformer = classifier.transformer
        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        clf = make_pipeline(vectorizer, transformer, classify)
        print cross_val_score(clf, datasets, dataset.target, cv=cv)
        print 'test_pipline_cross_val_score done!'
        print '-' * 70
        
        
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(SvmClassifierTest('test_loadfiles'))
    suite.addTest(SvmClassifierTest('test_bayesClassifier_GaussianNB'))
    suite.addTest(SvmClassifierTest('test_chi2'))
    suite.addTest(SvmClassifierTest('test_tfidf'))
    suite.addTest(SvmClassifierTest('test_CountVectorizer'))
    suite.addTest(SvmClassifierTest('test_jieba'))
    suite.addTest(SvmClassifierTest('test_compareTextAB'))
    #suite.addTest(SvmClassifierTest('test_testClassifier'))
    suite.addTest(SvmClassifierTest('test_pipline_cross_val_score'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')