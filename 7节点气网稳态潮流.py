# -*- coding: utf-8 -*-
"""
基于统一能路的7节点天然气网络稳态潮流计算程序 (Open source)
confirmed
"""

__author__ = 'Chen Binbin'

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from contextlib import contextmanager


@contextmanager
def context(event):
    t0 = time.time()
    print('[{}] {} starts ...'.format(time.strftime('%Y-%m-%d %H:%M:%S'), event))
    yield
    print('[{}] {} ends ...'.format(time.strftime('%Y-%m-%d %H:%M:%S'), event))
    print('[{}] {} runs for {:.2f} s'.format(time.strftime('%Y-%m-%d %H:%M:%S'), event, time.time()-t0))


with context('读取数据与处理'):
    pipe_table = pd.read_excel('./7节点气网稳态data.xls', sheet_name='Branch')
    node_table = pd.read_excel('./7节点气网稳态data.xls', sheet_name='Node')
    numpipe = len(pipe_table)  # 支路数
    numnode = len(node_table)  # 节点数
    l = pipe_table['长度(km)'].values * 1e3  # 长度，m
    d = pipe_table['管径(mm)'].values / 1e3  # 管径，m
    lam = pipe_table['粗糙度'].values  # 摩擦系数
    cp = pipe_table['压气机(MPa)'].values * 1e6  # 支路增压，Pa
    c = 340  # 声速
    Apipe = np.pi*d**2/4  # 管道截面积
    v = np.ones(numpipe)*5  # 流速基值


with context('基于能路的潮流计算'):
    MaxIter = 100  # 最大迭代次数
    err = []  # 误差记录
    vb = [v.copy()]  # 基值记录
    for itera in range(MaxIter):
        # 支路参数
        Rg = [lam[i]*v[i]/Apipe[i]/d[i] for i in range(numpipe)]
        Lg = [1/Apipe[i] for i in range(numpipe)]
        Cg = [Apipe[i]/c**2 for i in range(numpipe)]
        Ug = [-lam[i]*v[i]**2/2/c**2/d[i] for i in range(numpipe)]
        
        # 支路导纳矩阵
        Yb1, Yb2, Zb, Ub = [], [], [], []
        f = 0  # 稳态，只取零频率分量
        for i in range(numpipe):
            Z, Y = Rg[i], 0
            za = np.cosh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i]) - Ug[i]/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
            za = za*np.exp(-Ug[i]*l[i]/2)
            zb = -2*Z/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
            zb = zb*np.exp(-Ug[i]*l[i]/2)
            zc = -2*Y/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
            zc = zc*np.exp(-Ug[i]*l[i]/2)
            zd = np.cosh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i]) + Ug[i]/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
            zd = zd*np.exp(-Ug[i]*l[i]/2)
            
            Yb1.append((za*zd-zb*zc-za)/zb)  # 稳态计算中，接地支路不起作用
            Yb2.append((1-zd)/zb)  # 稳态计算中，接地支路不起作用
            Zb.append(-zb)
            Ub.append(1-za*zd+zb*zc)
        yb, ub = np.diag(1/np.array(Zb)), np.diag(Ub)
        
        # 节点-支路关联矩阵
        A = np.zeros([numnode, numpipe])
        Ap = np.zeros([numnode, numpipe])
        for row in pipe_table.iterrows():
            A[int(row[1][1])-1, row[0]] = 1
            A[int(row[1][2])-1, row[0]] = -1
            Ap[int(row[1][1])-1, row[0]] = 1
            Ap[int(row[1][2])-1, row[0]] = 0
        
        # 节点导纳矩阵
        Yg_ = np.matmul(np.matmul(A, yb), A.T) - np.matmul(np.matmul(np.matmul(A, yb), ub), Ap.T)
        
        # 节点分类
        fix_G = node_table[node_table['节点类型']=='定注入'].index.values
        fix_p = node_table[node_table['节点类型']=='定压力'].index.values
        Yg_11 = Yg_[fix_G][:,fix_G]
        Yg_12 = Yg_[fix_G][:,fix_p]
        Yg_21 = Yg_[fix_p][:,fix_G]
        Yg_22 = Yg_[fix_p][:,fix_p]
        assert np.linalg.cond(Yg_11)<1e5  #　确认矩阵不奇异
        
        # 形成广义节点注入向量(给定)　与　节点压力向量(给定)
        Gn_1 = node_table[node_table['节点类型']=='定注入']['注入 (kg/s)'].values.reshape([-1,1])  # kg/s
        Gn_1 -= np.matmul(np.matmul(A[fix_G,:], yb), cp.reshape([-1,1]))
        pn2 = node_table[node_table['节点类型']=='定压力']['气压 (MPa)'].values.reshape([-1,1]) * 1e6  # Pa
            
        # 求解零频率网络方程
        pn1 = np.matmul(np.linalg.inv(Yg_11), (Gn_1 - np.matmul(Yg_12, pn2))).real
        Gn_2 = (np.matmul(Yg_21, pn1) + np.matmul(Yg_22, pn2))
        Gn_2 += np.matmul(np.matmul(A[fix_p,:], yb), cp.reshape([-1,1]))
        
        # 计算失配误差
        p = []
        pn1, pn2 = pn1.reshape(-1).tolist(), pn2.reshape(-1).tolist()
        for node in node_table['节点类型'].values:
            p.append(pn1.pop(0) if node=='定注入' else pn2.pop(0))
        p = np.array(p).reshape([-1,1])
        I = np.matmul(A.T, p).reshape(-1) + cp.reshape(-1) - (np.array(Ub).reshape([-1,1])*np.matmul(Ap.T, p)).reshape(-1)
        for i in range(numpipe):
            I[i] = abs(I[i]*np.diag(yb)[i]/Apipe[i]/(np.matmul(Ap.T, p).reshape(-1)[i])*c**2)
        err.append(np.linalg.norm(I-v))
        print('第%d次迭代，失配误差为%.5f'%(itera+1, err[-1]))
        
        # 修正基值
        v += (I-v)*0.5
        vb.append(v.copy())
        if err[-1]<1e-3:
            break


with context('可视化'):
    vis = 0
    if vis:
        plt.figure(1)
        plt.plot(np.array(vb))
        plt.figure(2)
        plt.plot(err)
        plt.yscale('log')
        print(p)
        print(Gn_2)
