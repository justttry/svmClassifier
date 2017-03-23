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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
import jieba
from numpy import *
import unittest
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
def loadfiles(dirs):
    """
    加载数据
    Parameter:
    Return:
    """
    return load_files(dirs)

########################################################################
class DecisionClassifier(object):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, classifier, k='all'):
        """Constructor"""
        #基本分类器
        self.classifier = classifier
        #词频生成函数
        self.vectorizer = CountVectorizer(min_df=1)
        #TFIDF生成函数
        self.transformer = TfidfTransformer(smooth_idf=False)
        #特征选择数
        self.k = k 
        self.featureselect = SelectKBest(chi2, k=k)
        
    #----------------------------------------------------------------------
    def predict(self, X):
        """
        预测类别
        Parameter:
        X:被预测矩阵
        Parameter:
        y:预测类
        """
        tfidfs = self.transform(X)
        return self.classifier.predict(tfidfs)
    
    #----------------------------------------------------------------------
    def fit_transform(self, X, y):
        """"""
        counts = self.vectorizer.fit_transform(X)
        newcounts = self.featureselect.fit_transform(counts, y)
        tfidfs = self.transformer.fit_transform(newcounts, y)
        self.classifier.fit(tfidfs, y)
        return tfidfs
    
    #----------------------------------------------------------------------
    def fit(self, X, y):
        """"""
        self.fit_transform(X, y)
        
    #----------------------------------------------------------------------
    def transform(self, X):
        """"""
        counts = self.vectorizer.transform(X)
        newcounts = self.featureselect.transform(counts)
        return self.transformer.transform(newcounts)
    
    #----------------------------------------------------------------------
    def score(self, X, y):
        """"""
        tfidfs = self.transform(X)
        return self.classifier.score(tfidfs, y)


########################################################################
class DecisionClassifierTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_loadfiles(self):
        dirs = './test_file2'
        datasets = loadfiles(dirs)
        print 'test_loadfiles done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_pipline_cross_val_score(self):
        """"""
        #加载数据
        dataset = load_files('./test_file2')
        #对数据进行分词处理
        datasets = []
        for i in dataset.data:
            datasets.append(' '.join([j for j in jieba.cut(i)]))        #训练数据
        classifier = DecisionClassifier(DecisionTreeClassifier(), k=1000)
        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        clf = make_pipeline(classifier)
        print cross_val_score(clf, datasets, dataset.target, cv=cv)
        print 'test_pipline_cross_val_score done!'
        print '-' * 70
        
    #----------------------------------------------------------------------
    def test_figure_k_acurracy(self):
        """"""
        #加载数据
        dataset = load_files('./test_file2')
        #对数据进行分词处理
        datasets = []
        for i in dataset.data:
            datasets.append(' '.join([j for j in jieba.cut(i)]))        #训练数据
        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        ks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 
              3000, 5000, 10000, 20000, 'all']
        accuracys = []
        for k in ks:
            classifier = DecisionClassifier(DecisionTreeClassifier(), k=k)
            clf = make_pipeline(classifier)
            accuracys.append(\
                average(cross_val_score(clf, datasets, dataset.target, cv=cv)))
        fig, ax = plt.subplots()
        ax.scatter(range(len(ks)), accuracys)
        ax.set_xlabel('k')
        ax.set_ylabel('accuracy')
        plt.show()
        print 'test_figure_k_acurracy done!'
        print '-' * 70
        
        
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(DecisionClassifierTest('test_loadfiles'))
    suite.addTest(DecisionClassifierTest('test_pipline_cross_val_score'))
    suite.addTest(DecisionClassifierTest('test_figure_k_acurracy'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')