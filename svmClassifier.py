#encoding:UTF-8

from sklearn.datasets import load_files
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


########################################################################
class SvmClassifierTest(unittest.TestCase):
    """"""

    #----------------------------------------------------------------------
    def test_loadfiles(self):
        dirs = './test_file2'
        datasets = loadfiles(dirs)
        print 'test_loadfiles done!'
        
#----------------------------------------------------------------------
def suite():
    """"""
    suite = unittest.TestSuite()
    suite.addTest(SvmClassifierTest('test_loadfiles'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')