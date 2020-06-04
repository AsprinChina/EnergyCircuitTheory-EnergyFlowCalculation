# -*- coding: utf-8 -*-
"""
基于统一能路的7节点天然气网络动态潮流计算程序 (Open source)
confirmed
"""

__author__ = 'Chen Binbin'

import time
import numpy as np
import pandas as pd
from cmath import phase
from scipy.fftpack import fft
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
    # 原始数据
    pipe_table = pd.read_excel('./7节点气网动态data.xls', sheet_name='Branch')
    node_table = pd.read_excel('./7节点气网动态data.xls', sheet_name='Node')
    load_table = pd.read_excel('./7节点气网动态data.xls', sheet_name='Load')
    pres_table = pd.read_excel('./7节点气网动态data.xls', sheet_name='Pressure')
    # 数据初步处理
    numpipe = len(pipe_table)  # 支路数
    numnode = len(node_table)  # 节点数
    l = pipe_table['长度(km)'].values * 1e3  # 长度，m
    d = pipe_table['管径(mm)'].values / 1e3  # 管径，m
    lam = pipe_table['粗糙度'].values  # 摩擦系数
    cp = pipe_table['压气机(MPa)'].values * 1e6  # 支路增压，Pa
    c = 340  # 声速
    Apipe = np.pi*d**2/4  # 管道截面积    
    # 节点分类
    fix_G = node_table[node_table['节点类型']=='定注入'].index.values
    fix_p = node_table[node_table['节点类型']=='定压力'].index.values
    fix_p = np.append(fix_p, numnode)  # 最后一个节点，大地节点，是定压力节点    
    # 节点-支路关联矩阵
    A = np.zeros([numnode, numpipe])
    A0 = np.zeros([numnode+1, numpipe*3])  # 最后一行是地节点
    Ap = np.zeros([numnode, numpipe])
    Ap0 = np.zeros([numnode+1, numpipe*3])  # 最后一行是地节点
    for row in pipe_table.iterrows():
        A[int(row[1][1])-1, row[0]] = 1
        A[int(row[1][2])-1, row[0]] = -1
        A0[int(row[1][1])-1, row[0]*3] = 1
        A0[int(row[1][2])-1, row[0]*3] = -1
        A0[int(row[1][1])-1, row[0]*3+1] = 1
        A0[-1, row[0]*3+1] = -1
        A0[int(row[1][2])-1, row[0]*3+2] = 1
        A0[-1, row[0]*3+2] = -1
        Ap[int(row[1][1])-1, row[0]] = 1
        Ap[int(row[1][2])-1, row[0]] = 0
        Ap0[int(row[1][1])-1, row[0]*3] = 1
        Ap0[int(row[1][2])-1, row[0]*3] = 0
        Ap0[int(row[1][1])-1, row[0]*3+1] = 1
        Ap0[-1, row[0]*3+1] = 0
        Ap0[int(row[1][2])-1, row[0]*3+2] = 1
        Ap0[-1, row[0]*3+2] = 0

            
with context('动态潮流计算'):
    v = np.array([5.0]*6)  # 流速基值，平启动
    for itera in range(100):
        # 确定时域分布参数
        Rg = [lam[i]*v[i]/Apipe[i]/d[i] for i in range(numpipe)]
        Lg = [1/Apipe[i] for i in range(numpipe)]
        Cg = [Apipe[i]/c**2 for i in range(numpipe)]
        Ug = [-lam[i]*v[i]**2/2/c**2/d[i] for i in range(numpipe)]
        
        Node_Injection = load_table.fillna(0).values  # kg/s
        Node_Pressure = pres_table.fillna(0).values * 1e6  # Pa
        
        # TD_E: Time Domain Encourage (节点数行*时间点数列)
        TD_E_G = []
        TD_E_p = []
        for row in node_table.iterrows():
            TD_E_G.append(Node_Injection[:, row[0]])
            TD_E_p.append(Node_Pressure[:, row[0]])
        # 增加地节点
        TD_E_G.append(np.zeros(TD_E_G[-1].shape))
        TD_E_p.append(np.zeros(TD_E_p[-1].shape))
        TD_E_G = np.array(TD_E_G)
        TD_E_p = np.array(TD_E_p)
        
        # 原始时域激励数据，5min一个点，144个点，共12小时
        interpolation = 30  # interpolation倍插值，300/interpolation s一个点
        TD_E_G = TD_E_G.repeat(interpolation, axis=1)
        TD_E_p2 = np.zeros([numnode+1, 360*12])
        for i in range(numnode+1):
            TD_E_p2[i,:] = np.interp(np.linspace(10, 3600*12, 360*12),  # 插成10s一个点
                                     np.linspace(300, 3600*12, 12*12),  # 原来300s一个点
                                     TD_E_p[i,:])
        TD_E_p = TD_E_p2

        # 在0时刻之前叠加一段稳态激励，边值转初值
        k = 0.5  # 12*0.5 = 6, 叠加6小时历史边界条件
        initialization = int(TD_E_G.shape[1]*k)  # 插值
        TD_E_G = np.concatenate((TD_E_G[:,0].reshape([-1,1]).repeat(initialization,axis=1), TD_E_G), axis=1)
        TD_E_p = np.concatenate((TD_E_p[:,0].reshape([-1,1]).repeat(initialization,axis=1), TD_E_p), axis=1)
        TD_GL = np.zeros([numpipe, TD_E_G.shape[1]])  # 线路流量
        
        # FD_E: Frequency Domain Encourage (节点数行*频域分量数列)
        FD_E_G = []
        FD_E_p = [] 
        nf = 100*2  # 保留的频率分量数
        nt = TD_E_G.shape[1]  # 时域总长度
        fr = 1/(12*3600*(k+1))  # 频率分辨率（frequency resolution）
        FD_GL = np.zeros([numpipe, nf], dtype='complex_')
        FD_GL_FROM = np.zeros([numpipe, nf], dtype='complex_')
        FD_GL_TO = np.zeros([numpipe, nf], dtype='complex_')
        for row in node_table.iterrows():
            if row[1]['节点类型'] == '定压力':
                Gf = np.zeros(nf, dtype='complex_')
                pf = (fft(TD_E_p[row[0],:])/nt*2)[:nf]
                pf[0] /= 2
            elif row[1]['节点类型'] == '定注入':
                pf = np.zeros(nf, dtype='complex_')
                Gf = (fft(TD_E_G[row[0],:])/nt*2)[:nf]
                Gf[0] /= 2
            else:
                assert 0
            FD_E_G.append(Gf)
            FD_E_p.append(pf)
        # 增加地节点
        FD_E_G.append(np.zeros(nf, dtype='complex_'))
        FD_E_p.append(np.zeros(nf, dtype='complex_'))
        FD_E_G = np.array(FD_E_G)
        FD_E_p = np.array(FD_E_p)
    
        # 单频能路计算, 遍历所有频率分量
        for fi in range(FD_E_G.shape[1]):
            f = fi*fr
            # 计算各支路的频域集总参数（pi型等值）
            Yb1, Yb2, Zb, Ub = [], [], [], []
            for i in range(numpipe):
                Z = Rg[i] + complex(0,1)*2*np.pi*f*Lg[i]
                Y = complex(0,1)*2*np.pi*f*Cg[i]
                za = np.cosh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i]) - Ug[i]/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
                za = za*np.exp(-Ug[i]*l[i]/2)
                zb = -2*Z/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
                zb = zb*np.exp(-Ug[i]*l[i]/2)
                zc = -2*Y/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
                zc = zc*np.exp(-Ug[i]*l[i]/2)
                zd = np.cosh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i]) + Ug[i]/np.sqrt(Ug[i]**2+4*Z*Y)*np.sinh(np.sqrt(Ug[i]**2+4*Z*Y)/2*l[i])
                zd = zd*np.exp(-Ug[i]*l[i]/2)
                Yb1.append((za*zd-zb*zc-za)/zb)  # 动态潮流中，接地支路起作用
                Yb2.append((1-zd)/zb)  # 动态潮流中，接地支路起作用
                Zb.append(-zb)
                Ub.append(1-za*zd+zb*zc)
    
            # 形成支路导纳矩阵
            yb = np.diag(np.array([[1/Zb[i], Yb1[i], Yb2[i]] for i in range(numpipe)]).reshape(-1))
            # 形成支路受控电压源矩阵
            ub = np.diag(np.array([[Ub[i], 0, 0] for i in range(numpipe)]).reshape(-1))
            # 支路气压源向量
            cpexp = np.array([[cp[i], 0, 0] for i in range(len(cp))]).reshape([-1,1]) if fi==0 else np.zeros([18,1])
            # 形成广义节点导纳矩阵
            Yg_ = np.matmul(np.matmul(A0, yb), A0.T) - np.matmul(np.matmul(np.matmul(A0, yb), ub), Ap0.T)
            Yg_11 = Yg_[fix_G][:,fix_G]
            Yg_12 = Yg_[fix_G][:,fix_p]
            Yg_21 = Yg_[fix_p][:,fix_G]
            Yg_22 = Yg_[fix_p][:,fix_p]
            
           # 求解网络方程
            FD_E_p[fix_G, fi] = np.matmul(np.linalg.inv(Yg_11), ((FD_E_G[fix_G, fi]-np.matmul(np.matmul(A0[fix_G,:], yb), cpexp).reshape(-1)) - np.matmul(Yg_12, FD_E_p[fix_p, fi])))
            FD_E_G[fix_p, fi] = np.matmul(Yg_21, FD_E_p[fix_G, fi]) + np.matmul(Yg_22, FD_E_p[fix_p, fi]) + np.matmul(np.matmul(A0[fix_p,:], yb), cpexp).reshape(-1)
            
            # 节点状态变量转支路状态变量
            FD_GL[:,fi] = (np.matmul(A.T, FD_E_p[:-1,fi]).reshape(-1) - (np.array(Ub).reshape([-1])*np.matmul(Ap.T, FD_E_p[:-1,fi]) - (cp if fi==0 else np.zeros(numpipe))).reshape(-1))/np.array(Zb)

            Afrom, Ato = A.copy(), A.copy()
            Afrom[Afrom<0] = 0
            Ato[Ato>0] = 0
            # 首端流量
            FD_GL_FROM[:,fi] = (np.matmul(A.T, FD_E_p[:-1,fi]).reshape(-1) - (np.array(Ub).reshape([-1])*np.matmul(Ap.T, FD_E_p[:-1,fi]) - (cp if fi==0 else np.zeros(numpipe))).reshape(-1))/np.array(Zb) + np.matmul(Afrom.T, FD_E_p[:-1, fi].reshape([-1,1])).reshape(-1)*np.array(Yb1)
            # 末端流量
            FD_GL_TO[:,fi] = (np.matmul(A.T, FD_E_p[:-1,fi]).reshape(-1) - (np.array(Ub).reshape([-1])*np.matmul(Ap.T, FD_E_p[:-1,fi]) - (cp if fi==0 else np.zeros(numpipe))).reshape(-1))/np.array(Zb) + np.matmul(Ato.T, FD_E_p[:-1, fi].reshape([-1,1])).reshape(-1)*np.array(Yb2)

        # from FD_E to TD_E
        ts = np.linspace(10, nt*10, nt)
        # 定压力节点的注入
        for fi in range(nf):
            for node in fix_p:
                TD_E_G[node, :] += abs(FD_E_G[node, fi])*np.cos(2*np.pi*fi*fr*ts + phase(FD_E_G[node, fi]))
        # 定注入节点的压力
        for fi in range(nf):
            for node in fix_G:
                TD_E_p[node, :] += abs(FD_E_p[node, fi])*np.cos(2*np.pi*fi*fr*ts + phase(FD_E_p[node, fi]))
        # 支路流量
        TD_GL_FROM = np.zeros(TD_GL.shape)
        TD_GL_TO = np.zeros(TD_GL.shape)
        for fi in range(nf):
            for branch in range(numpipe):
                TD_GL[branch, :] += abs(FD_GL[branch, fi])*np.cos(2*np.pi*fi*fr*ts + phase(FD_GL[branch, fi]))
                TD_GL_FROM[branch,:] += abs(FD_GL_FROM[branch,fi])*np.cos(2*np.pi*fi*fr*ts + phase(FD_GL_FROM[branch, fi]))
                TD_GL_TO[branch,:] += abs(FD_GL_TO[branch,fi])*np.cos(2*np.pi*fi*fr*ts + phase(FD_GL_TO[branch, fi]))
        # 支路流速
        TD_v = TD_GL / np.matmul(Ap.T, TD_E_p[:-1,:]) / Apipe.reshape([-1,1]) * c**2
        # 修正基值
        print('第%d次迭代，失配误差为%.5f'%(itera+1, np.linalg.norm(v - np.average(abs(TD_v),axis=1))))
        if np.linalg.norm(v - np.average(abs(TD_v),axis=1))<1e-1:
            break
        v += (np.average(abs(TD_v), axis=1)-v)*0.6


with context('可视化与输出'):
    vis = 1
    if vis:
        TD_E_G *= 1  # 从kg/s转换回kg/s
        TD_E_p /= 1e6  # 从Pa转换回MPa
        plt.figure(1)
        plt.plot(TD_E_G.T)
        plt.figure(2)
        plt.plot(TD_E_p[:-1].T)
        # print(TD_E_G)
        # print(TD_E_p)
    """ 导出潮流数据
    """
    # 1条支路/1个节点 对应 1行数据，时间间隔为10s，6小时（历史边界）+12小时（计算潮流）
    if False:  # 根据需要打开
        pd.DataFrame(TD_GL_FROM).to_csv('./潮流数据/支路首端流量.csv', header=0, index=0)
        pd.DataFrame(TD_GL_TO).to_csv('./潮流数据/支路末端流量.csv', header=0, index=0)
        pd.DataFrame(TD_E_G).to_csv('./潮流数据/节点注入.csv', header=0, index=0)
        pd.DataFrame(TD_E_p).to_csv('./潮流数据/节点压力.csv', header=0, index=0)

