import numpy as np
from guptri_py import guptri, kcf_blocks
import sys

kx = sys.argv[1]
ky = sys.argv[2]
hx = sys.argv[3]
q = sys.argv[4]
datadir = sys.argv[5]
NormMat = np.load(datadir+"kx{}ky{}NormMat.npy".format(kx, ky))
HamiMat = np.load(datadir+"kx{}ky{}HamiMat.npy".format(kx, ky))
NormMat = NormMat.reshape(np.prod(NormMat.shape[:5]), np.prod(NormMat.shape[5:]))
HamiMat = HamiMat.reshape(np.prod(HamiMat.shape[:5]), np.prod(HamiMat.shape[5:]))
NormMat = NormMat/(2*11+2)**2
HamiMat = 2*HamiMat/(2*11+2)**3/(2*11+1)

Es = []
S, T, P, Q, kstr = guptri(HamiMat, NormMat, epsu=1e-4)
# print("S=", S)
# print("T=", T)
kcfBs = kcf_blocks(kstr)
# print("kcfBs=", kcfBs)
nrow = kcfBs[0,2]
ncol = kcfBs[1,2]
accurow = np.sum(kcfBs[0,:2])
# print("accurow=", accurow)
accucol = np.sum(kcfBs[1,:2])
# print("accucol=", accucol)
assert nrow == ncol
for i in range(nrow):
    # print("i=", i)
    # print("S[accurow+i,accucol+i]=", S[accurow+i,accucol+i])
    # print("T[accurow+i,accucol+i]=", T[accurow+i,accucol+i])
    Es.append(S[accurow+i,accucol+i] / T[accurow+i,accucol+i])
        
Es = np.array(Es)
print("Es=", (2*11+2)*(2*11+1)*(Es+2.810809581874137))

Es = (2*11+2)*(2*11+1)*(Es+2.810809581874137)
for i,e in reversed(list(enumerate(Es))):
    if np.abs(e.imag)/np.abs(e) > 0.1:
        # raise ValueError("Es is not real")
        print("remove ES {}", i)
        # Es = np.delete(Es, i)
        Es[i] = np.abs(Es[i])

# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# print(np.angle(Es))
# print(np.abs(Es))
# ax.plot(np.angle(Es), np.abs(Es), 'o', color='black', markersize=2)
# # ax.set_rmax(2)
# # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
# # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
# ax.grid(True)
# ax.set_title("A line plot on a polar axis", va='bottom')
# plt.show()

Es = Es.real
with open(datadir+"guptri_excitedE.txt", "a") as f:
    f.write("#kx={}, ky={}, hx={}, q={}\n".format(kx, ky, hx, q))
    f.write(" ".join(str(e) for e in Es))
    f.write("\n")