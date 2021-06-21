# -*- coding: utf-8 -*-
"""
UEC-EnergyFlowCalculation-package
"""

__author__ = 'Chen Binbin'

import time
import numpy as np
import pandas as pd
from cmath import phase
from scipy.fftpack import fft
from matplotlib import pyplot as plt
from contextlib import contextmanager


class Network(object):
    def __init__(self):
        pass

    @staticmethod
    def _fft1d(tds, nf):
        tds = tds.reshape(-1)
        fds = fft(tds) / len(tds) * 2
        fds[0] /= 2
        return fds[:nf]

    @staticmethod
    def _formA(table, nNode, nBranch, fVal=1, tVal=-1):
        A = np.zeros((nNode, nBranch), dtype=np.int32)
        for _, row in table.iterrows():
            A[row['from node'] - 1, row['id'] - 1] = fVal
            A[row['to node'] - 1, row['id'] - 1] = tVal
        return A


class DHS(Network):
    def __init__(self, excelFile):
        tb1 = pd.read_excel('./6节点热网动态data.xls', sheet_name='Node').fillna(0)
        tb2 = pd.read_excel('./6节点热网动态data.xls', sheet_name='Branch')
        tb3 = pd.read_excel('./6节点热网动态data.xls', sheet_name='Device', header=None, index_col=0)
        tb4 = pd.read_excel('./6节点热网动态data.xls', sheet_name='Dynamic')

    def steadyHydraulicFlowCal(self):
        pass

    def steadyThermalFlowCal(self):
        pass

    def dynamicThermalFlowCal(self):
        pass

    def _buildHydraulicUEC(self):
        pass

    def _buildThermalUEC(self):
        pass


class NGS(Network):
    def __init__(self):
        pass

    def steadyFlowCal(self):
        pass

    def dynamicFlowCal(self):
        pass

    def _builtGasUEC(self):
        pass

