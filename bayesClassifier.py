#encoding:UTF-8

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
class SvmClassifierTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_loadfiles(self):
        dirs = './test_file2'
        datasets = loadfiles(dirs)
        print 'test_loadfiles done!'
        
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
        
    #----------------------------------------------------------------------
    def test_CountVectorizer(self):
        """"""
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(min_df=1)
        corpus = ["我 来到 北京 清华大学",  
                  "他 来到 了 网易 杭研 大厦",  
                  "小明 硕士 毕业 与 中国 科学院 小明",  
                  "我 爱 北京 天安门"] 
        X = vectorizer.fit_transform(corpus)        
        for i in vectorizer.get_feature_names():
            print i
        print X.toarray()
        print 'test_CountVectorizer done!'
        
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
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')