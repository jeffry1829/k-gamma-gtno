# python excitation_hei_stat_ori.py --GLOBALARGS_dtype complex128 --bond_dim 2 --chi 8 --size 11 --seed 123 --j2 0. --instate ex-j20D2chi8c4v_state.json
# python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --seed 123 --j2 0. --Kz 1.5 --num_h 10
# python SMA.py --GLOBALARGS_dtype complex128 --bond_dim 4 --chi 8 --size 11 --Kz 1.5 --num_h 10
# python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim 2 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python SMA_stored_mat_withP.py --hx 2.5 --GLOBALARGS_dtype complex128 --bond_dim 2 --size 12 --statefile D=2TFIM_output_state.json --chi 8 --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cpu
# python graph_withP.py
import matplotlib.pylab as pylab
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys
import context
import time
import torch
import argparse
import config as cfg
from ipeps.ipeps import *
from ctm.generic.env import *
from ctm.generic.rdm import *
# from ctm.generic import ctmrg_ex
from ctm.generic import ctmrg
from ctm.generic.ctm_projectors import *
# from Stat_ori import *
from Norm_ori import *
# from Hami_ori import *
from Localsite_Hami_ori import *
# from Test import *
# from models import j1j2
# from models import ising
from groups.pg import *
import groups.su2 as su2
from optim.ad_optim_lbfgs_mod import optimize_state
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import scipy.io
import unittest
import logging
from ctm.generic import rdm
import os
log = logging.getLogger(__name__)

tStart = time.time()

# from models import aniso_k
# from GTNOs import *

datadir = "./"
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
print("datadir = "+sys.argv[1]+"")

DefineTitleAndTick = False
if len(sys.argv) >= 3:
    if sys.argv[2] == "True":
        DefineTitleAndTick = True
    print("DefineTitleAndTick = "+str(sys.argv[2])+"")

TitleEval = ""
if len(sys.argv) >= 4:
    TitleEval = sys.argv[3]
    print("TitleEval = "+sys.argv[3]+"")

TickEval = ""
if len(sys.argv) >= 5:
    TickEval = sys.argv[4]
    print("TickEval = "+sys.argv[4]+"")
##################################################################################################################################
USE_LOG_SW = True
##################################################################################################################################


params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (10, 10),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)
# font = {'family': 'Times New Roman',
#         # 'weight': 'bold',
#         'size': 12}

# plt.rc('font', **font)

# Plot two vison

if os.path.isfile(datadir+"TV.txt") == True:
    cnt = -1
    Es = []
    TVs = []
    points = []
    values = []

    x = []
    y = []
    z = []
    with open(datadir+"TV.txt", "r") as f:
        for line in f:
            # if cnt>=1:
            #     break
            if line[0] == "#":
                cnt += 1
                continue
            Es.append(float(line.split()[0]))
            TVs.append(float(line.split()[1]))
            points.append((cnt, float(line.split()[0])))
            values.append(float(line.split()[1]))
            x.append(cnt)
            y.append(float(line.split()[0]))
            z.append(float(line.split()[1]))

    from matplotlib import cm
    ax = plt.subplot()
    plt.grid(True, linestyle='-', color='0.75')
    # scatter with colormap mapping to z value
    plt.scatter(x, y, s=20, c=z, marker='o', cmap=cm.get_cmap('viridis'))
    plt.title('(momentum, energy) to Kzz')
    # plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
    #                             r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
    plt.xticks([0, 4, 8, 16], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
               r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
    # plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
    #            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
    # plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
    # plt.xticks([0,12,18,24],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
    if DefineTitleAndTick:
        plt.title("(momentum, energy) to Kzz " + TitleEval)
        eval("plt.xticks("+TickEval+")")
    plt.xlabel('momentum')
    plt.ylabel('excitation energy')
    plt.gcf().set_size_inches(6, 6)
    plt.colorbar()
    plt.show()

if os.path.isfile(datadir+"XXA.txt") == True:
    cnt = -1
    Es = []
    XXAs = []

    x = []
    y = []
    with open(datadir+"XXA.txt", "r") as f:
        for line in f:
            # if cnt>=1:
            #     break
            if line[0] == "#":
                cnt += 1
                continue
            Es.append(float(line.split()[0]))
            XXAs.append(float(line.split()[1]))
            x.append(cnt)
            y.append(float(line.split()[1]))

    from matplotlib import cm
    ax = plt.subplot()
    plt.plot(x, y, 'o')
    # plt.grid(True, linestyle='-', color='0.75')
    # scatter with colormap mapping to z value
    # plt.scatter(x, y, s=20, c=z, marker='o', cmap=cm.get_cmap('viridis'))
    plt.title('(momentum, energy) to XXA')
    # plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
    #                             r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
    plt.xticks([0, 4, 8, 16], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
               r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
    # plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
    #            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
    # plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
    # plt.xticks([0,12,18,24],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
    if DefineTitleAndTick:
        plt.title("momentum to XXA " + TitleEval)
        eval("plt.xticks("+TickEval+")")
    plt.xlabel('momentum')
    plt.ylabel('XXA')
    # plt.gcf().set_size_inches(6, 6)
    # plt.colorbar()
    plt.show()

if os.path.isfile(datadir+"XIIXA.txt") == True:
    cnt = -1
    Es = []
    XIIXAs = []

    x = []
    y = []
    with open(datadir+"XIIXA.txt", "r") as f:
        for line in f:
            # if cnt>=1:
            #     break
            if line[0] == "#":
                cnt += 1
                continue
            Es.append(float(line.split()[0]))
            XIIXAs.append(float(line.split()[1]))
            x.append(cnt)
            y.append(float(line.split()[1]))

    from matplotlib import cm
    ax = plt.subplot()
    plt.plot(x, y, 'o')
    # plt.grid(True, linestyle='-', color='0.75')
    # scatter with colormap mapping to z value
    # plt.scatter(x, y, s=20, c=z, marker='o', cmap=cm.get_cmap('viridis'))
    plt.title('(momentum, energy) to XIIXA')
    # plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
    #                             r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
    plt.xticks([0, 4, 8, 16], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
               r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
    # plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
    #            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
    # plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
    # plt.xticks([0,12,18,24],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
    if DefineTitleAndTick:
        plt.title("momentum to XIIXA " + TitleEval)
        eval("plt.xticks("+TickEval+")")
    plt.xlabel('momentum')
    plt.ylabel('XIIXA')
    # plt.gcf().set_size_inches(6, 6)
    # plt.colorbar()
    plt.show()


# Plot spectral weight

cnt = -1
Es = []
SWs = []
singleSW = []
grpSWs = []
points = []
values = []

x = []
y = []
z = []
with open(datadir+"SW.txt", "r") as f:
    for line in f:
        if line[0] == "#":
            cnt += 1
            if cnt > 0:
                grpSWs.append(singleSW)
            singleSW = []
            continue
        Es.append(float(line.split()[0]))
        singleSW.append(float(line.split()[1]))
        if USE_LOG_SW:
            SWs.append(np.log(float(line.split()[1])))
        else:
            SWs.append(float(line.split()[1]))
        points.append((cnt, float(line.split()[0])))
        if USE_LOG_SW:
            values.append(np.log(float(line.split()[1])))
        else:
            values.append(float(line.split()[1]))
        x.append(cnt)
        y.append(float(line.split()[0]))
        if USE_LOG_SW:
            z.append(np.log(float(line.split()[1])))
        else:
            z.append(float(line.split()[1]))

# # x is (kx,ky)
# # y is w
# grid_step = 1
# grid_x, grid_y = np.mgrid[0:cnt:grid_step, np.amin(Es):
#                           np.amax(Es):grid_step]
# points2 = points.copy()
# values2 = values.copy()
# for _x in np.arange(0, cnt+1, grid_step):
#     for _y in np.arange(math.floor(np.amin(Es)), math.ceil(np.amax(Es)), grid_step):
#         points2.append((_x, _y))
#         val = 0
#         for i, pt in enumerate(points):
#             distanceptsq = (pt[0]-_x)**2+(pt[1]-_y)**2
#             sigma = 0.1
#             val += values[i]*np.exp(-distanceptsq /
#                                     (2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
#         # values2.append(0)
#         values2.append(val)
# grid_z2 = griddata(points2, values2, (grid_x, grid_y), method='linear')
# # plt.subplot(221)
# # plt.imshow(grid_z2.T, extent=(0,1,np.amin(Es),np.amax(Es)), origin='lower')
# plt.subplot()
# im = plt.imshow(grid_z2.T, extent=(
#     0, cnt, np.amin(Es), np.amax(Es)), origin='lower', aspect='auto')
# plt.title('spectral weight')
# # plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
# #            r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
# # plt.xticks([0, 6, 12, 24], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
# #            r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
# plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
#            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# # plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# # plt.xticks([0,6,9,12],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
# plt.xlabel('momentum')
# plt.ylabel('excitation energy')
# plt.gcf().set_size_inches(6, 6)
# ax = plt.gca()
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# plt.show()

ax = plt.subplot()
plt.grid(True, linestyle='-', color='0.75')
# scatter with colormap mapping to z value
plt.scatter(x, y, s=20, c=z, marker='o', cmap=cm.get_cmap('viridis'))
plt.title('(momentum, energy) to spectral weight')
# plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
#            r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
plt.xticks([0, 4, 8, 16], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
                           r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
# plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
#            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# plt.xticks([0,6,9,12],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
if DefineTitleAndTick:
    plt.title("(momentum, energy) to spectral weight "+TitleEval)
    eval("plt.xticks("+TickEval+")")
plt.xlabel('momentum')
plt.ylabel('excitation energy')
plt.gcf().set_size_inches(6, 6)
plt.colorbar()
plt.show()

# fig, ax = plt.subplots(figsize=(6.5,5), dpi = 100)
# plt.pcolormesh(points[0], points[1], values, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, shading= 'gouraud', antialiased=True, data=None)
# plt.colorbar()
# plt.show()

plt.subplot()

sumSW = []
cntlst = []
cnt = -1
for i in range(len(grpSWs)):
    sumSW.append(sum(grpSWs[i]))
    cnt += 1
    cntlst.append(cnt)
ax = plt.subplot()
plt.grid(True, linestyle='-', color='0.75')
plt.plot(cntlst, sumSW, 'o')
plt.title('sum spectral weight')
# plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
#            r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
plt.xticks([0, 4, 8, 16], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
                           r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
# plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
#            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# plt.xticks([0,6,9,12],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
if DefineTitleAndTick:
    plt.title("sum spectral weight "+TitleEval)
    eval("plt.xticks("+TickEval+")")
plt.xlabel('momentum')
plt.ylabel('sum spectral weight')
plt.gcf().set_size_inches(6, 6)
# plt.colorbar()
plt.show()

# Es = []
# cntlst = []
# cnt = -1
# with open(datadir+"excitedE.txt", "r") as f:
#     for line in f:
#         if line[0] == "#":
#             cnt += 1
#             continue
#         cntlst.append(cnt)
#         Es.append(float(line.split()[0]))

# plt.xticks([0,6,12,24],[r'$M(\pi,0)$',r'$\Gamma(0,0)$',r'$K(\pi,\frac{\pi}{2})$',r'$M(\pi,0)$'])
# # plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# # plt.xticks([0,6,9,12],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
# plt.xlabel('momentum')
# plt.ylabel('lowest-excitation energy')
# plt.plot(cntlst, Es, 'o')
# plt.show()

# Es = []
# cntlst = []
# cnt = -1
# with open(datadir+"eigN.txt", "r") as f:
#     for line in f:
#         if line[0] == "#":
#             cnt += 1
#             continue
#         for e in line.split():
#             cntlst.append(cnt)
#             Es.append(float(e))

# plt.plot(cntlst, Es, 'o')
# plt.show()

cnt = -1
SSFs = []
cntlst = []

with open(datadir+"SSF.txt", "r") as f:
    for line in f:
        if line[0] == "#":
            cnt += 1
            continue
        SSFs.append(float(line.split()[0]))
        cntlst.append(cnt)
ax = plt.subplot()
plt.grid(True, linestyle='-', color='0.75')
plt.plot(cntlst, SSFs, 'o')
plt.title('static structure factor')
# plt.xticks([0, 8, 16, 32], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
#            r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
plt.xticks([0, 4, 8, 16], [r'$M(\pi,0)$', r'$\Gamma(0,0)$',
                           r'$K(\pi,\frac{\pi}{2})$', r'$M(\pi,0)$'])
# plt.xticks([0,6,12,24],[r'$M(\pi,0)$',r'$\Gamma(0,0)$',r'$K(\pi,\frac{\pi}{2})$',r'$M(\pi,0)$'])
# plt.xticks([0, 6, 9, 12, 18, 21], [r'$M(\pi,0)$', r'$X(\pi,\pi)$',
#            r'$S(\frac{\pi}{2},\frac{\pi}{2})$', r'$\Gamma(0,0)$', r'$M(\pi,0)$', r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# plt.xticks([0,12,18,24,36,42],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$',r'$M(\pi,0)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$'])
# plt.xticks([0,6,9,12],[r'$M(\pi,0)$',r'$X(\pi,\pi)$',r'$S(\frac{\pi}{2},\frac{\pi}{2})$',r'$\Gamma(0,0)$'])
if DefineTitleAndTick:
    plt.title("static structure factor "+TitleEval)
    eval("plt.xticks("+TickEval+")")
plt.xlabel('momentum')
plt.ylabel('static structure factor')
plt.gcf().set_size_inches(6, 6)
# plt.colorbar()
plt.show()
