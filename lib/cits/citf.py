from ..cits.cit_m import MCITest
from ..cits.cit_kr import KRCITest
from ..cits.cit_sic import SICTest
from ..cits.cit_rbf import RBFCITest
from ..cits.cit_naive import NaiveCITest

class CITFactory:

    MCIT    = 'mcit'
    KRCIT   = 'krcit'
    NAIVE   = 'naive'
    SIC     = 'sic'
    RBF     = 'rbf'

    @staticmethod
    def get_cit(name, seed):
        if name == CITFactory.MCIT:
            return MCITest(seed)
        elif name == CITFactory.KRCIT:
            return KRCITest(seed)
        elif name == CITFactory.SIC:
            return SICTest(seed)
        elif name == CITFactory.RBF:
            return RBFCITest(seed)
        elif name == CITFactory.NAIVE:
            return NaiveCITest(seed)
        else:
            raise "%s cit class not defined" % name