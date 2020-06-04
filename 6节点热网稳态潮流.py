# -*- coding: utf-8 -*-
"""
基于统一能路的6节点供热网络稳态潮流计算程序 (Open source)
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


with context('数据读取与处理'):
    tb1 = pd.read_excel('./6节点热网稳态data.xls', sheet_name='Node').fillna(0)
    tb2 = pd.read_excel('./6节点热网稳态data.xls', sheet_name='Branch')
    tb3 = pd.read_excel('./6节点热网稳态data.xls', sheet_name='Device', header=None, index_col=0)
    # 水力参数
    L = tb2['length'].values * 1e3
    D = tb2['diameter'].values
    lam = tb2['fraction'].values
    npipes, nnodes = len(tb2), len(tb1)
    As = np.array([np.pi*d**2/4 for d in D])
    mb = np.ones(npipes) * 50  # 基值平启动
    rho = 1000
    Ah = np.zeros([nnodes, npipes], dtype=np.int32)
    for i,row in tb2.iterrows():
        Ah[row['from node']-1, i] = 1
        Ah[row['to node']-1, i] = -1
    fix_p = np.where(tb1.type1.values=='定压力')[0]
    fix_G = np.where(tb1.type1.values=='定注入')[0]
    # 热力参数
    c = 4200
    miu = tb2.disspation.values
    
    
with context('稳态水力计算'):
    err = []  # 失配误差记录
    mbs = [mb.copy()]  # 流量基值的迭代过程记录
    for itera in range(100):  # 最大迭代次数
        # 更新支路参数
        R = [lam[i]*mb[i]/rho/As[i]**2/D[i]*L[i] for i in range(npipes)]
        E = [-lam[i]*mb[i]**2/2/rho/As[i]**2/D[i]*L[i] for i in range(npipes)]
        # 追加各支路阀、泵的参数
        for i,row in tb2.iterrows():
            if row.pump > 0:
                kp1, kp2, kp3, w = tb3.loc['pump-%d'%int(row.pump),:]
                R[i] += -(2*kp1*mb[i]+kp2*w)
                E[i] += (kp1*mb[i]**2-kp3*w**2)
            if row.valve > 0:
                kv, _, _, _ = tb3.loc['valve-%d'%int(row.valve),:]
                R[i] += 2*kv*mb[i]**2
                E[i] -= -kv*mb[i]**2
        E = np.array(E).reshape([-1,1])
        yb = np.diag([1/Ri for Ri in R])
        Y = np.matmul(np.matmul(Ah, yb), Ah.T)
        Ygg = Y[fix_G][:,fix_G]
        Ygp = Y[fix_G][:,fix_p]
        Ypg = Y[fix_p][:,fix_G]
        Ypp = Y[fix_p][:,fix_p]
        pp = tb1['pressure(MPa)'].values[fix_p].reshape([1,1]) * 1e6
        G = tb1['injection(kg/s)'].values.reshape([-1,1]) + np.matmul(np.matmul(Ah, yb), E)
        Gg = G[fix_G,:]
        assert np.linalg.cond(Ygg)<1e5  # 确认导纳矩阵非奇异
        pg = np.matmul(np.linalg.inv(Ygg), (Gg - np.matmul(Ygp, pp)))
        pn = np.concatenate((pp, pg), axis=0)
        Gb = np.matmul(yb, (np.matmul(Ah.T, pn) - E))
        err.append(np.linalg.norm(Gb.reshape(-1) - mb))
        mb = mb*0.2 + Gb.reshape(-1)*0.8
        mbs.append(mb.copy())
        print('第%d次迭代，失配误差为%.5f'%(itera+1, err[-1]))
        if err[-1] < 1e-3:
            print('水力稳态潮流计算迭代%d次后收敛。'%(itera+1))
            break


with context('稳态热力计算'):
    m = list(Gb.reshape(-1))  # 各支路流量
    A = Ah  # 水力、热力共享一个节点支路关联矩阵
    Af = np.zeros(A.shape)
    Af[A>0] = 1
    At_ = np.zeros(A.shape)
    At_[A<0] = 1
    for i in range(A.shape[1]):
        for j in range(A.shape[0]):
            At_[j,i] *= m[i]
    for i in range(A.shape[0]):
        if sum(At_[i,:])==0:
            continue
        At_[i,:] /= sum(At_[i,:])
    # 稳态广义激励
    Tin = tb1['T(Celsius)'].values
    E = tb2.deltaT.values
    w = 0  # 仅直流分量Z[j]
    Rh = np.array([miu[j]/c**2/m[j]**2 for j in range(A.shape[1])])
    # Lh = np.array([rho*As[j]/c/m[j]**2 for j in range(A.shape[1])])
    Z = Rh  # + complex(0,1)*w*Lh
    K = np.diag([np.exp(-c*m[j]*Z[j]*L[j]) for j in range(A.shape[1])])    
    
    _m_ = np.linalg.inv(np.eye(A.shape[1]) - np.matmul(np.matmul(K, Af.T), At_))
    Tt = np.matmul(_m_, np.matmul(np.matmul(K, Af.T), Tin) + E)
    Tf = np.matmul(np.linalg.inv(K), Tt - E)
    
    
with context('可视化与输出'):
    vis = 0
    if vis:
        plt.figure(1)
        plt.plot(np.array(mbs))
        plt.figure(2)
        plt.plot(err)
        plt.yscale('log')
        # print(m)
        # print(pn)
        # print(Tf)
        # print(Tt)
        
    
    