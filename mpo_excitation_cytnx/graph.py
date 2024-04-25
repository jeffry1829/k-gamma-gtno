import matplotlib.pyplot as plt
import sys
datadir = "./"
if len(sys.argv) == 2:
    print("datadir = "+sys.argv[1]+"")
    datadir = sys.argv[1]
    
Jx = -1.0
Jy = -1.0
Jz = -1.0
h = 0.0
Dmps = 8
import numpy as np
xs = []
ys = []
with open(datadir+"Jx{}Jy{}Jz{}h{}Dmps{}.txt".format(Jx,Jy,Jz,h,Dmps), "r") as f:
    for line in f:
        if line[0] == "#":
            continue
        