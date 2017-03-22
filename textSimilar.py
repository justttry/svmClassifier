#encoding:UTF-8

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

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


if __name__ == '__main__':
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