import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
data_ = np.loadtxt("kitaev_kz21.txt")
# data = np.loadtxt("kitaev_kz21_polar.txt")
topo = np.loadtxt("topo.dat")
# data_ = np.loadtxt("kitaev_out_polar.txt")
h = topo[1122:1173, 1]
I3 = topo[1122:1173, 2]
# print(data_)
K = 1
Kz = 2.1

print((K**4/(32*2.1**2))**0.5)

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
# plt.plot(data[:, 3], data[:, 4], marker = 'o', markersize = 5, label = "iPEPS D=4 (from polar)")
plt.plot(data_[:, 3], data_[:, 4], marker='o',
         markersize=5, lw=0, label=r'$ \langle e \rangle$')
# plt.axvline(x = (K**4/(32*Kz**2))**0.5, linestyle = '-', color = 'k',lw = 1, label = r'$h_c \approx 0.084$')
plt.axvline(x=(K**4/(32*Kz**2))**0.5, linestyle='-', color='k', lw=1)
# plt.ylim(-0.31,-0.29)
plt.xlabel('h')
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
x = np.flip(data_[:, 3])
print(x)
y = np.flip(data_[:, 4])
y_spl = UnivariateSpline(x, y, s=0, k=3)
x_range = np.linspace(x[0], x[-1], 100)
y_spl_1d = y_spl.derivative(n=2)
plt.plot(x_range, y_spl_1d(x_range), marker='o', markersize=0,
         lw=4, label=r'$d^2 \langle e \rangle/dh^2$')
# plt.axvline(x = (K**4/(32*Kz**2))**0.5, linestyle = '-', color = 'k',lw = 1, label = r'$h_c \approx 0.084$')
plt.xlabel('h')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
plt.plot(h, -I3, marker='o', markersize=5, lw=0, label="ED 24-sites")
# plt.plot(data[:, 0], -data[:, -1], marker = 'o', markersize = 5, label = "iPEPS D=4")
plt.plot(data_[:, 3], -data_[:, -1], marker='o',
         markersize=5, lw=0,  label="iPEPS D=4")
plt.axhline(y=np.log(2), linestyle='-', color='b', lw=1, label=r'$\ln 2$')
# plt.axvline(x = (K**4/(32*Kz**2))**0.5, linestyle = '-', color = 'k',lw = 1, label = r'$h_c \approx 0.071$')
plt.legend()
plt.show()

Wp = np.loadtxt("Wp.dat")
h = Wp[1122:1173, 1]
wp = Wp[1122:1173, 2]

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
plt.plot(h, wp, marker='o', markersize=5, lw=0, label="ED 24-sites")
# plt.plot(data[:, 0], data[:, 3], marker = 'o', markersize = 5, label = "iPEPS D=4")
plt.plot(data_[:, 3], data_[:, 6], marker='o',
         markersize=5, lw=0, label="iPEPS D=4")
# plt.axvline(x = (K**4/(32*Kz**2))**0.5, linestyle = '-', color = 'k',lw = 1, label = r'$h_c \approx 0.084$')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
# plt.plot(data[:, 0], data[:, -2], marker = 'o', markersize = 5, label = "iPEPS D=4")
plt.plot(data_[:, 3], data_[:, -2], marker='o',
         markersize=5, lw=0, label="iPEPS D=4 (Qzz)")
plt.plot(data_[:, 3], data_[:, -4], marker='o',
         markersize=5, lw=0, label="iPEPS D=4 (Qxx)")
# plt.axvline(x = (K**4/(32*Kz**2))**0.5, linestyle = '-', color = 'k',lw = 1, label = r'$h_c \approx 0.071$')
plt.legend()
plt.show()
