import torch
import groups.su2 as su2
from ctm.generic import rdm
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import corrf_c4v
import config as cfg
import numpy as np
from math import sqrt
from ctm.generic import corrf
import itertools
from GTNOs import *
from ipeps.ipeps import *


def _cast_to_real(t):
    return t.real if t.is_complex() else t


def Svn(eigvals):
    S = 0
    for i in eigvals:
        k = i.item()
        if k < 1e-13:
            k = 1e-13
        S -= k*np.log(k)
    return S


class KITAEV():
    def __init__(self, kx=1.0, ky=1.0, kz=1.0, h=0.0, global_args=cfg.global_args):
        r"""
        :param hx: transverse field
        :param q: plaquette interaction 
        :param global_args: global configuration
        :type hx: float
        :type q: float
        :type global_args: GLOBALARGS

        Build Ising Hamiltonian in transverse field with plaquette interaction

        .. math:: H = - \sum_{<i,j>} h2_{<i,j>} + q\sum_{p} h4_p - h_x\sum_i h1_i

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over 
        all plaquettes `p`, and the last sum runs over all sites::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = 4S^z_i S^z_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        * :math:`h4_p  = 16S^z_i S^z_j S^z_k S^z_l` where `i,j,k,l` labels the sites of a plaquette::

            p= i---j
               |   |
               k---l 

          and the indices of `h4` correspond to :math:`s_is_js_ks_l;s'_is'_js'_ks'_l`

        * :math:`h1_i  = 2S^x_i`
        """
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = 2  # honeycomb to square lattice
        self.h = h
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.x_inter, self.y_inter, self.z_inter, self.local_h = self.get_h_2x2()
        # self.zz, self.xx, self.yy, self.hp = self.get_h()

        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        I = torch.eye(2, dtype=self.dtype, device=self.device)
        Z = 2*s2.SZ()
        X = s2.SP()+s2.SM()
        Y = -(s2.SP()-s2.SM())*1j

        self.XI = torch.einsum('ij,ab->iajb', X, I)
        self.IX = torch.einsum('ij,ab->iajb', I, X)
        self.IY = torch.einsum('ij,ab->iajb', I, Y)
        self.YI = torch.einsum('ij,ab->iajb', Y, I)
        self.ZI = torch.einsum('ij,ab->iajb', Z, I)
        self.IZ = torch.einsum('ij,ab->iajb', I, Z)
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        # Note that these should act on a 2x2 rdm

        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        # id2= id2.view(2,2,2,2).contiguous()
        Sx = s2.SP()+s2.SM()
        Sy = -(s2.SP()-s2.SM())*1j
        SzSz = torch.einsum('ij,ab->iajb', 2*s2.SZ(), 2*s2.SZ())
        # I
        # XX
        # I
        SxSxII = torch.einsum('ij,ab,cd,ef->iacejbdf', Sx, Id, Id, Sx)
        # I
        # Y
        # Y
        # I
        SySyII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sy, Sy, Id)
        hp = -SzSz - SxSxII - SySyII

        return SzSz, SxSxII, SySyII, hp

    def get_h_2x2(self):
        # Note that these should act on a 2x2 rdm

        s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Sz = 2*s2.SZ()
        Sx = s2.SP()+s2.SM()
        Sy = -(s2.SP()-s2.SM())*1j

        XX = torch.einsum('ij,ab->iajb', Sx, Sx)
        # ZY = torch.einsum('ij,ab->iajb',Sz,Sy)

        # I
        # XX
        # I
        # XYII = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sx, Sy, Id)
        # YXII = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sy, Sx, Id)
        # ZZII = (torch.einsum('ij,ab,cd,ef->iacejbdf',Sz, Id, Id, Sz)+torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sz, Sz, Id))/2
        # ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf',Sz, Id, Id, Sz)
        ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sz, Sz, Id)
        # I
        # Y
        # Y
        # I
        # XZII = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sx, Sz, Id)
        # ZXII = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sz, Sx, Ids)
        # YYII = (torch.einsum('ij,ab,cd,ef->iacejbdf',Sy, Id, Id, Sy)+torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sy, Sy, Id))/2
        YYII = torch.einsum('ij,ab,cd,ef->iacejbdf', Sy, Id, Id, Sy)

        H = 0.5*(torch.einsum('ij,ab->iajb', Sx+Sy+Sz, Id) +
                 torch.einsum('ij,ab->iajb', Id, Sx+Sy+Sz))
        # XZII = torch.einsum('ij,ab,cd,ef->iacejbdf',Sx, Id, Id, Sz)
        # ZXII = torch.einsum('ij,ab,cd,ef->iacejbdf',Sz, Id, Id, Sx)
        # return YZII+ZYII, XZII+ZXII, XY+YX
        # return YZ+ZY, XZII+ZXII, XYII+YXII
        return XX, YYII, ZZII, H
    # def get_h(self):
    #     s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
    #     Id= torch.eye(2,dtype=self.dtype,device=self.device)
    #     # id2= id2.view(2,2,2,2).contiguous()
    #     Sx = s2.SP()+s2.SM()
    #     Sy = -(s2.SP()-s2.SM())*1j
    #     SzSz = torch.einsum('ij,ab->iajb',2*s2.SZ(),2*s2.SZ())
    #     SxSx = torch.einsum('ij,ab->iajb',Sx, Sx)
    #     SySy = torch.einsum('ij,ab->iajb',Sy, Sy)
    #     hp = -SzSz - SxSx - SySy

    #     return SzSz, SxSx, SySy, hp

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"] = 2*s2.SZ()
        obs_ops["sp"] = 2*s2.SP()
        obs_ops["sm"] = 2*s2.SM()
        return obs_ops

    def energy_1x1(self, state, env):
        # s0 s1
        # s2 s3
        # Note that by wraping with tensor will result in req_grad related errors!
        rdm1x1 = rdm.rdm1x1((0, 0), state, env)  # s0,s1,s2,s3,s0',s1',s2',s3'
        # print(rdm2x2.type(torch.complex64))
        energy_per_bond = torch.einsum('ij,ij', rdm1x1.type(
            self.dtype), self.hp.reshape(4, 4).type(self.dtype))/3
        # print(energy_per_bond)
        energy_per_bond = _cast_to_real(energy_per_bond)
        return energy_per_bond

    # def energy_2x2(self,state,env):
    #     ## s0 s1
    #     ## s2 s3
    #     ## Note that by wraping with tensor will result in req_grad related errors!
    #     rdm2x2 = rdm.rdm2x2((0,0),state,env) # s0,s1,s2,s3,s0',s1',s2',s3'
    #     # print(rdm2x2.type(torch.complex64))
    #     energy_per_site = torch.einsum('iabcjabc,ij',rdm2x2.type(self.dtype),-self.zz.reshape(4,4).type(self.dtype))
    #     energy_per_site += torch.einsum('ijabmnab,ijmn',rdm2x2.type(self.dtype),-self.xx.reshape(4,4,4,4).type(self.dtype))
    #     # energy_per_site = torch.einsum('ijabmnab,ijmn',rdm2x2.type(self.dtype),-self.xx.reshape(4,4,4,4).type(self.dtype))
    #     energy_per_site += torch.einsum('iajbmanb,ijmn',rdm2x2.type(self.dtype),-self.yy.reshape(4,4,4,4).type(self.dtype))
    #     energy_per_site = _cast_to_real(energy_per_site)
    #     return energy_per_site/8

    # def eval_obs(self,state,env):
    #     r"""
    #     :param state: wavefunction
    #     :param env: CTM environment
    #     :type state: IPEPS
    #     :type env: ENV
    #     :return:  expectation values of observables, labels of observables
    #     :rtype: list[float], list[str]

    #     Computes the following observables in order

    #         1. :math:`\langle 2S^z \rangle,\ \langle 2S^x \rangle` for each site in the unit cell

    #     """
    #     obs= dict()
    #     with torch.no_grad():
    #         for coord,site in state.sites.items():
    #             rdm1x1= rdm.rdm1x1(coord,state,env)
    #             for label,op in self.obs_ops.items():
    #                 obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
    #             obs[f"sx{coord}"]= 0.5*(obs[f"sp{coord}"] + obs[f"sm{coord}"])

    #         for coord,site in state.sites.items():
    #             rdm2x1= rdm.rdm2x1(coord,state,env)
    #             rdm1x2= rdm.rdm1x2(coord,state,env)
    #             rdm2x2= rdm.rdm2x2(coord,state,env)
    #             SzSz2x1= torch.einsum('ijab,ijab',rdm2x1,self.h2)
    #             SzSz1x2= torch.einsum('ijab,ijab',rdm1x2,self.h2)
    #             SzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
    #             obs[f"SzSz2x1{coord}"]= _cast_to_real(SzSz2x1)
    #             obs[f"SzSz1x2{coord}"]= _cast_to_real(SzSz1x2)
    #             obs[f"SzSzSzSz{coord}"]= _cast_to_real(SzSzSzSz)

    #     # prepare list with labels and values
    #     obs_labels= [f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sx"]))]
    #     obs_labels+= [f"SzSz2x1{coord}" for coord in state.sites.keys()]
    #     obs_labels+= [f"SzSz1x2{coord}" for coord in state.sites.keys()]
    #     obs_labels+= [f"SzSzSzSz{coord}" for coord in state.sites.keys()]
    #     obs_values=[obs[label] for label in obs_labels]
    #     return obs_values, obs_labels

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"] = 2*s2.SZ()
        obs_ops["sp"] = 2*s2.SP()
        obs_ops["sm"] = 2*s2.SM()
        return obs_ops

    def energy_2x2(self, state, env):
        # s0 s1
        # s2 s3
        # Note that by wraping with tensor will result in req_grad related errors!
        rdm2x2 = rdm.rdm2x2((0, 0), state, env)  # s0,s1,s2,s3,s0',s1',s2',s3'
        # print("rdm shape = ", rdm2x2.shape)
        # energy_per_site = torch.einsum('iabcjabc,ij',rdm2x2.type(self.dtype),self.z_inter.reshape(4,4).type(self.dtype))
        Ex = torch.einsum('iabcjabc,ji', rdm2x2.type(
            self.dtype), self.x_inter.reshape(4, 4).type(self.dtype))
        # Ex_ = Ex
        energy_per_site = self.kx*Ex
        # Ey = torch.einsum('ijabmnab,ijmn',rdm2x2.type(self.dtype),self.x_inter.reshape(4,4,4,4).type(self.dtype))
        Ey = torch.einsum('iajbmanb,mnij', rdm2x2.type(
            self.dtype), self.y_inter.reshape(4, 4, 4, 4).type(self.dtype))
        # Ey_ = torch.einsum('aibjambn,mnij',rdm2x2.type(self.dtype),self.y_inter.reshape(4,4,4,4).type(self.dtype))
        energy_per_site += self.ky*Ey
        # Ez = torch.einsum('iajbmanb,ijmn',rdm2x2.type(self.dtype),self.y_inter.reshape(4,4,4,4).type(self.dtype))
        Ez = torch.einsum('ijabmnab,mnij', rdm2x2.type(
            self.dtype), self.z_inter.reshape(4, 4, 4, 4).type(self.dtype))
        # Ez_ = torch.einsum('abijabmn,mnij',rdm2x2.type(self.dtype),self.z_inter.reshape(4,4,4,4).type(self.dtype))
        energy_per_site += self.kz*Ez
        energy_per_site += self.h*torch.einsum('iabcjabc,ji', rdm2x2.type(
            self.dtype), -self.local_h.reshape(4, 4).type(self.dtype))*4
        energy_per_site = _cast_to_real(energy_per_site)

        # print("x: {},y: {},z: {}".format(Ex.real,Ey.real,Ez.real))
        # print("x_: {},y_: {},z_: {}".format(Ex_.real,Ey_.real,Ez_.real))

        # with torch.no_grad():
        #     s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
        #     Id= torch.eye(2,dtype=self.dtype,device=self.device)
        #     Z = 2*s2.SZ()
        #     X = s2.SP()+s2.SM()
        #     Y = -(s2.SP()-s2.SM())*1j
        #     # IZ = torch.einsum('ij,ab->iajb',Id,Z).reshape(4,4).type(self.dtype)
        #     # ZI = torch.einsum('ij,ab->iajb',Z,Id).reshape(4,4).type(self.dtype)
        #     # XY = torch.einsum('ij,ab->iajb',X,Y).reshape(4,4).type(self.dtype)
        #     # YX = torch.einsum('ij,ab->iajb',Y,X).reshape(4,4).type(self.dtype)
        #     II = torch.einsum('ij,ab->iajb',Id,Id).reshape(4,4).type(self.dtype)
        #     XI = torch.einsum('ij,ab->iajb',Id,X).reshape(4,4).type(self.dtype)
        #     IX = torch.einsum('ij,ab->iajb',X,Id).reshape(4,4).type(self.dtype)
        #     YZ = torch.einsum('ij,ab->iajb',Y,Z).reshape(4,4).type(self.dtype)
        #     ZY = torch.einsum('ij,ab->iajb',Z,Y).reshape(4,4).type(self.dtype)
        #     Wp = torch.einsum('abcdlmno,al,bm,cn,do',rdm2x2.type(self.dtype),ZY,XI,IX,YZ)
        #     Wp = _cast_to_real(Wp)
        #     print("Wp = ", Wp)
        #     Wp = torch.einsum('abcdlmno,al,bm,cn,do',rdm2x2.type(self.dtype),IX,ZY,YZ,XI)
        #     Wp = _cast_to_real(Wp)
        #     print("Wp = ", Wp)
        #     Wp = torch.einsum('abcdlmno,al,bm,cn,do',rdm2x2.type(self.dtype),YZ,IX,XI,ZY)
        #     Wp = _cast_to_real(Wp)
        #     print("Wp = ", Wp)
        #     Wp = torch.einsum('abcdlmno,al,bm,cn,do',rdm2x2.type(self.dtype),XI,YZ,ZY,IX)
        #     Wp = _cast_to_real(Wp)
        #     print("Wp = ", Wp)
        return energy_per_site*(1/8)

    # def energy_2x2_2site(self,state,env): # Description copied from peps-torch, not neccessary true situation
    #     r"""
    #     :param state: wavefunction
    #     :param env: CTM environment
    #     :type state: IPEPS
    #     :type env: ENV
    #     :return: energy per site
    #     :rtype: float
    #     We assume iPEPS with 2x1 unit cell containing two tensors A, B. We can
    #     tile the square lattice in two ways::
    #         BIPARTITE           STRIPE
    #         A B A B             A B A B
    #         B A B A             A B A B
    #         A B A B             A B A B
    #         B A B A             A B A B
    #     Taking reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster with indexing
    #     of sites as follows :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::

    #         s0--s1
    #         |   |
    #         s2--s3
    #     and without assuming any symmetry on the indices of individual tensors following
    #     set of terms has to be evaluated in order to compute energy-per-site::

    #            0
    #         1--A--3
    #            2

    #         Ex.1 unit cell A B, with BIPARTITE tiling
    #             A3--1B, B3--1A, A, B, A3  , B3  ,   1A,   1B
    #                             2  0   \     \      /     /
    #                             0  2    \     \    /     /
    #                             B  A    1A    1B  A3    B3

    #         Ex.2 unit cell A B, with STRIPE tiling
    #             A3--1A, B3--1B, A, B, A3  , B3  ,   1A,   1B
    #                             2  0   \     \      /     /
    #                             0  2    \     \    /     /
    #                             A  B    1B    1A  B3    A3
    #     """
    #     # A3--1B   B3  1A
    #     # 2 \/ 2   2 \/ 2
    #     # 0 /\ 0   0 /\ 0
    #     # B3--1A & A3  1B

    #     # A3--1B   B3--1A
    #     # 2 \/ 2   2 \/ 2
    #     # 0 /\ 0   0 /\ 0
    #     # A3--1B & B3--1A
    #     energy_per_site=torch.zeros(1,dtype=self.dtype,device=self.device)
    #     for coord in state.sites.keys():
    #         print("coordinate:")
    #         print(coord)
    #         tmp_rdm= rdm.rdm2x2(coord,state,env)
    #         # energy_nn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nn)
    #         # energy_nnn += torch.einsum('ijklabcd,ijklabcd',tmp_rdm,self.h2x2_nnn)

    #         Ex = torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.x_inter.reshape(4,4).type(self.dtype))
    #         # Ex2 = torch.einsum('abciabcj,ji',tmp_rdm.type(self.dtype),self.x_inter.reshape(4,4).type(self.dtype))
    #         # print(Ex)
    #         # print(energy_per_site)
    #         energy_per_site += Ex
    #         # energy_per_site += Ex2
    #         Ey = torch.einsum('iajbmanb,mnij',tmp_rdm.type(self.dtype),self.y_inter.reshape(4,4,4,4).type(self.dtype))
    #         # Ey2 = torch.einsum('aibjambn,mnij',tmp_rdm.type(self.dtype),self.y_inter.reshape(4,4,4,4).type(self.dtype))
    #         energy_per_site += Ey
    #         # energy_per_site += Ey2
    #         Ez = torch.einsum('ijabmnab,mnij',tmp_rdm.type(self.dtype),self.z_inter.reshape(4,4,4,4).type(self.dtype))
    #         # Ez2 = torch.einsum('abijabmn,mnij',tmp_rdm.type(self.dtype),self.z_inter.reshape(4,4,4,4).type(self.dtype))
    #         energy_per_site += Ez
    #         # energy_per_site += Ez2
    #         # print("x1: {}, x2: {},y1: {}, y2: {},z1: {}, z2: {}".format(Ex,Ex2,Ey,Ey2,Ez,Ez2))
    #         print("Ex: {}, Ey: {}, Ez: {}".format(Ex.real,Ey.real,Ez.real))

    #         mag_A1 = [] # [mx, my, mz]
    #         mag_A2 = []
    #         mag_B1 = []
    #         mag_B2 = []
    #         mag_A1.append(torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.XI.reshape(4,4).type(self.dtype)).real.item())
    #         mag_A1.append(torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.YI.reshape(4,4).type(self.dtype)).real.item())
    #         mag_A1.append(torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.ZI.reshape(4,4).type(self.dtype)).real.item())
    #         mag_A2.append(torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.IX.reshape(4,4).type(self.dtype)).real.item())
    #         mag_A2.append(torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.IY.reshape(4,4).type(self.dtype)).real.item())
    #         mag_A2.append(torch.einsum('iabcjabc,ji',tmp_rdm.type(self.dtype),self.IZ.reshape(4,4).type(self.dtype)).real.item())
    #         mag_B1.append(torch.einsum('aibcajbc,ji',tmp_rdm.type(self.dtype),self.XI.reshape(4,4).type(self.dtype)).real.item())
    #         mag_B1.append(torch.einsum('aibcajbc,ji',tmp_rdm.type(self.dtype),self.YI.reshape(4,4).type(self.dtype)).real.item())
    #         mag_B1.append(torch.einsum('aibcajbc,ji',tmp_rdm.type(self.dtype),self.ZI.reshape(4,4).type(self.dtype)).real.item())
    #         mag_B2.append(torch.einsum('aibcajbc,ji',tmp_rdm.type(self.dtype),self.IX.reshape(4,4).type(self.dtype)).real.item())
    #         mag_B2.append(torch.einsum('aibcajbc,ji',tmp_rdm.type(self.dtype),self.IY.reshape(4,4).type(self.dtype)).real.item())
    #         mag_B2.append(torch.einsum('aibcajbc,ji',tmp_rdm.type(self.dtype),self.IZ.reshape(4,4).type(self.dtype)).real.item())
    #         print("M_A1: {}\nM_A2: {}\nM_B1: {}\nM_B2: {}".format(mag_A1,mag_A2,mag_B1,mag_B2))
    #     # energy_per_site= 2.0*(self.j1*energy_nn/8.0 + self.j2*energy_nnn/4.0)
    #     # energy_per_site= _cast_to_real(energy_per_site)
    #     energy_per_site = _cast_to_real(energy_per_site)
    #     energy_per_site*=(1/16)
    #     #return -energy_per_site
    #     return -energy_per_site

    # def eval_obs(self,state,env):
    #     r"""
    #     :param state: wavefunction
    #     :param env: CTM environment
    #     :type state: IPEPS
    #     :type env: ENV
    #     :return:  expectation values of observables, labels of observables
    #     :rtype: list[float], list[str]

    #     Computes the following observables in order

    #         1. :math:`\langle 2S^z \rangle,\ \langle 2S^x \rangle` for each site in the unit cell

    #     """
    #     obs= dict()
    #     with torch.no_grad():
    #         for coord,site in state.sites.items():
    #             rdm1x1= rdm.rdm1x1(coord,state,env)
    #             for label,op in self.obs_ops.items():
    #                 obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
    #             obs[f"sx{coord}"]= 0.5*(obs[f"sp{coord}"] + obs[f"sm{coord}"])

    #         for coord,site in state.sites.items():
    #             rdm2x1= rdm.rdm2x1(coord,state,env)
    #             rdm1x2= rdm.rdm1x2(coord,state,env)
    #             rdm2x2= rdm.rdm2x2(coord,state,env)
    #             SzSz2x1= torch.einsum('ijab,ijab',rdm2x1,self.h2)
    #             SzSz1x2= torch.einsum('ijab,ijab',rdm1x2,self.h2)
    #             SzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
    #             obs[f"SzSz2x1{coord}"]= _cast_to_real(SzSz2x1)
    #             obs[f"SzSz1x2{coord}"]= _cast_to_real(SzSz1x2)
    #             obs[f"SzSzSzSz{coord}"]= _cast_to_real(SzSzSzSz)

    #     # prepare list with labels and values
    #     obs_labels= [f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sx"]))]
    #     obs_labels+= [f"SzSz2x1{coord}" for coord in state.sites.keys()]
    #     obs_labels+= [f"SzSz1x2{coord}" for coord in state.sites.keys()]
    #     obs_labels+= [f"SzSzSzSz{coord}" for coord in state.sites.keys()]
    #     obs_values=[obs[label] for label in obs_labels]
    #     return obs_values, obs_labels

    # def get_Q(self,state,env):

    #     rdm2x2 = rdm.rdm2x2((0,0),state,env) # s0,s1,s2,s3,s0',s1',s2',s3'

    #     Ex = torch.einsum('iabcjabc,ji',rdm2x2.type(self.dtype),self.x_inter.reshape(4,4).type(self.dtype))
    #     energy_per_site = Ex
    #     # Ey = torch.einsum('ijabmnab,ijmn',rdm2x2.type(self.dtype),self.x_inter.reshape(4,4,4,4).type(self.dtype))
    #     Ey = torch.einsum('iajbmanb,mnij',rdm2x2.type(self.dtype),self.y_inter.reshape(4,4,4,4).type(self.dtype))
    #     energy_per_site += Ey
    #     # Ez = torch.einsum('iajbmanb,ijmn',rdm2x2.type(self.dtype),self.y_inter.reshape(4,4,4,4).type(self.dtype))
    #     Ez = torch.einsum('ijabmnab,mnij',rdm2x2.type(self.dtype),self.z_inter.reshape(4,4,4,4).type(self.dtype))
    #     return XX

    # def get_m(self,state,env,bdtype):
    #     s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
    #     I= torch.eye(2,dtype=self.dtype,device=self.device)
    #     Z = 2*s2.SZ()
    #     X = s2.SP()+s2.SM()
    #     Y = -(s2.SP()-s2.SM())*1j

    #     XI = torch.einsum('ij,ab->iajb',X,I)
    #     IX = torch.einsum('ij,ab->iajb',I,X)
    #     IY = torch.einsum('ij,ab->iajb',I,Y)
    #     YI = torch.einsum('ij,ab->iajb',Y,I)
    #     ZI = torch.einsum('ij,ab->iajb',Z,I)
    #     IZ = torch.einsum('ij,ab->iajb',I,Z)

    #     if bdtype == "x":
    #         rdm1x1 = rdm.rdm1x1((0,0),state,env) # s0,s1,s2,s3,s0',s1',s2',s3'
    #         e1 = torch.einsum('ij,ji',rdm1x1.type(self.dtype),XI.reshape(4,4).type(self.dtype))
    #         e2 = torch.einsum('ij,ji',rdm1x1.type(self.dtype),IX.reshape(4,4).type(self.dtype))
    #     if bdtype == "y":
    #         rdm1x1 = rdm.rdm1x1((0,0),state,env) # s0,s1,s2,s3,s0',s1',s2',s3'
    #         e1 = torch.einsum('ij,ji',rdm1x1.type(self.dtype),YI.reshape(4,4).type(self.dtype))
    #         e2 = torch.einsum('ij,ji',rdm1x1.type(self.dtype),IY.reshape(4,4).type(self.dtype))
    #     if bdtype == "z":
    #         rdm1x1 = rdm.rdm1x1((0,0),state,env) # s0,s1,s2,s3,s0',s1',s2',s3'
    #         e1 = torch.einsum('ij,ji',rdm1x1.type(self.dtype),ZI.reshape(4,4).type(self.dtype))
    #         e2 = torch.einsum('ij,ji',rdm1x1.type(self.dtype),IZ.reshape(4,4).type(self.dtype))
    #     return _cast_to_real(e1), _cast_to_real(e2)

    # def get_Wp(self,state,env):
    #     with torch.no_grad():
    #         tmp_rdm = rdm.rdm2x2((0,0),state,env)
    #         s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
    #         Id= torch.eye(2,dtype=self.dtype,device=self.device)
    #         Z = 2*s2.SZ()
    #         X = s2.SP()+s2.SM()
    #         Y = -(s2.SP()-s2.SM())*1j
    #         # IZ = torch.einsum('ij,ab->iajb',Id,Z).reshape(4,4).type(self.dtype)
    #         # ZI = torch.einsum('ij,ab->iajb',Z,Id).reshape(4,4).type(self.dtype)
    #         # XY = torch.einsum('ij,ab->iajb',X,Y).reshape(4,4).type(self.dtype)
    #         # YX = torch.einsum('ij,ab->iajb',Y,X).reshape(4,4).type(self.dtype)
    #         XI = torch.einsum('ij,ab->iajb',Id,X).reshape(4,4).type(self.dtype)
    #         IX = torch.einsum('ij,ab->iajb',X,Id).reshape(4,4).type(self.dtype)
    #         YZ = torch.einsum('ij,ab->iajb',Y,Z).reshape(4,4).type(self.dtype)
    #         ZY = torch.einsum('ij,ab->iajb',Z,Y).reshape(4,4).type(self.dtype)
    #         Wp = torch.einsum('abcdlmno,al,bm,cn,do',tmp_rdm.type(self.dtype),IX,ZY,YZ,XI)
    #         # Wp = torch.einsum('abcdabcd',rdm2x2.type(self.dtype))
    #         Wp = _cast_to_real(Wp)
    #         # print("Wp = ", Wp)
    #     return [Wp.detach().clone().cpu().real.item()], ["Wp"]

    def eval_obs(self, state, env):
        obs_labels = ["Wp", "Wp2", "<X>", "<Y>", "<Z>",
                      "XX_z", "YY_z", "ZZ_z", "Qxx", "Qyy", "Qzz", "TEE"]
        with torch.no_grad():
            tmp_rdm = rdm.rdm2x2((0, 0), state, env)
            # tmp_rdm = torch.einsum('abcdijmn->acbdimjn',tmp_rdm)
            s2 = su2.SU2(2, dtype=self.dtype, device=self.device)
            Id = torch.eye(2, dtype=self.dtype, device=self.device)
            Z = 2*s2.SZ()
            X = s2.SP()+s2.SM()
            Y = -(s2.SP()-s2.SM())*1j
            XI = torch.einsum('ij,ab->iajb', X,
                              Id).reshape(4, 4).type(self.dtype)
            IX = torch.einsum('ij,ab->iajb', Id,
                              X).reshape(4, 4).type(self.dtype)
            YZ = torch.einsum('ij,ab->iajb', Y, Z).reshape(4,
                                                           4).type(self.dtype)
            ZY = torch.einsum('ij,ab->iajb', Z, Y).reshape(4,
                                                           4).type(self.dtype)

            # XY = torch.einsum('ij,ab->iajb',X,Y).reshape(4,4).type(self.dtype)
            # YX = torch.einsum('ij,ab->iajb',Y,X).reshape(4,4).type(self.dtype)
            # IZ = torch.einsum('ij,ab->iajb',Id,Z).reshape(4,4).type(self.dtype)
            # ZI = torch.einsum('ij,ab->iajb',Z,Id).reshape(4,4).type(self.dtype)

            # Wp = torch.einsum('abcdlmno,al,bm,cn,do',tmp_rdm.type(self.dtype),IX,ZY,YZ,XI)
            Wp = torch.einsum('abcdlmno,la,mb,nc,od',
                              tmp_rdm.type(self.dtype), IX, ZY, YZ, XI)
            Wp = _cast_to_real(Wp).detach().clone().cpu().real.item()

            # Wp2 = torch.einsum('abcdlmno,al,bm,cn,do',tmp_rdm.type(self.dtype),IX,YZ,ZY,XI)
            # Wp2 = _cast_to_real(Wp2).detach().clone().cpu().real.item()
            # Wp2 = torch.einsum('abcdlmno,al,bm,cn,do',tmp_rdm.type(self.dtype),IX,ZY,YZ,IX)
            Wp2 = torch.einsum('abcdlmno,la,mb,nc,od',
                               tmp_rdm.type(self.dtype), ZY, XI, IX, YZ)
            # Wp2 = torch.einsum('abcdlmno,al,bm,cn,do',tmp_rdm.type(self.dtype),IX,YZ,ZY,XI)
            Wp2 = _cast_to_real(Wp2).detach().clone().cpu().real.item()

            # Megnatization :
            oMx = (torch.einsum('ij,ab->iajb', X, Id) +
                   torch.einsum('ij,ab->iajb', Id, X))/2
            oMy = (torch.einsum('ij,ab->iajb', Y, Id) +
                   torch.einsum('ij,ab->iajb', Id, Y))/2
            oMz = (torch.einsum('ij,ab->iajb', Z, Id) +
                   torch.einsum('ij,ab->iajb', Id, Z))/2
            Mx = torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), oMx.reshape(
                4, 4).type(self.dtype)).detach().clone().cpu().real.item()
            Mz = torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), oMz.reshape(
                4, 4).type(self.dtype)).detach().clone().cpu().real.item()
            My = torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), oMy.reshape(
                4, 4).type(self.dtype)).detach().clone().cpu().real.item()

            # Q^{ab} for detecting the dimerization :
            x_inter_ = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, X, X, Id)
            y_inter_ = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Y, Y, Id)
            z_inter_ = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Z, Z, Id)
            XX_z = torch.einsum('ijabmnab,mnij', tmp_rdm.type(self.dtype), x_inter_.reshape(
                4, 4, 4, 4).type(self.dtype)).detach().clone().cpu().real.item()
            YY_z = torch.einsum('ijabmnab,mnij', tmp_rdm.type(self.dtype), y_inter_.reshape(
                4, 4, 4, 4).type(self.dtype)).detach().clone().cpu().real.item()
            ZZ_z = torch.einsum('ijabmnab,mnij', tmp_rdm.type(self.dtype), z_inter_.reshape(
                4, 4, 4, 4).type(self.dtype)).detach().clone().cpu().real.item()
            Qxx = XX_z - (1/3)*(XX_z+YY_z+ZZ_z)
            Qyy = YY_z - (1/3)*(XX_z+YY_z+ZZ_z)
            Qzz = ZZ_z - (1/3)*(XX_z+YY_z+ZZ_z)

            # Von neumann entropy test :
            # rdm1x1 = rdm.rdm1x1([0,0],state,env)
            # s = torch.diag(rdm1x1) # 2 Svx + 2 Svy

            # Kitaev-Preskill construction:
            rho_a = torch.einsum(
                'iabcjabc', tmp_rdm.type(self.dtype)).reshape(4, 4)
            rho_b = torch.einsum(
                'aibcajbc', tmp_rdm.type(self.dtype)).reshape(4, 4)
            rho_c = torch.einsum(
                'abimabjn->imjn', tmp_rdm.type(self.dtype)).reshape(16, 16)
            rho_ab = torch.einsum(
                'imabjnab->imjn', tmp_rdm.type(self.dtype)).reshape(16, 16)
            rho_bc = torch.einsum('aijkamnl->ijkmnl',
                                  tmp_rdm.type(self.dtype)).reshape(64, 64)
            rho_ac = torch.einsum('iajkmanl->ijkmnl',
                                  tmp_rdm.type(self.dtype)).reshape(64, 64)
            rho_abc = tmp_rdm.type(self.dtype).reshape(256, 256)

            # Sa = -torch.trace(torch.matmul(rho_a, torch.log(rho_a)))
            # Sb = -torch.trace(torch.matmul(rho_b, torch.log(rho_b)))
            # Sc = -torch.trace(torch.matmul(rho_c, torch.log(rho_c)))
            # Sab = -torch.trace(torch.matmul(rho_ab, torch.log(rho_ab)))
            # Sbc = -torch.trace(torch.matmul(rho_bc, torch.log(rho_bc)))
            # Sac = -torch.trace(torch.matmul(rho_ac, torch.log(rho_ac)))
            # Sabc = -torch.trace(torch.matmul(rho_abc, torch.log(rho_abc)))
            # TEE_ = Sa + Sb + Sc - Sab - Sbc - Sac + Sabc

            Sa = Svn(torch.linalg.eigvals(rho_a).real)
            Sb = Svn(torch.linalg.eigvals(rho_b).real)
            Sc = Svn(torch.linalg.eigvals(rho_c).real)
            Sab = Svn(torch.linalg.eigvals(rho_ab).real)
            Sbc = Svn(torch.linalg.eigvals(rho_bc).real)
            Sac = Svn(torch.linalg.eigvals(rho_ac).real)
            Sabc = Svn(torch.linalg.eigvals(rho_abc).real)

            TEE = Sa + Sb + Sc - Sab - Sbc - Sac + Sabc

            # print("\nSa: {}, Sb: {}, Sc: {}, Sab: {}, Sbc: {}, Sac: {}, Sabc: {}\n".format(Sa,Sb,Sc,Sab,Sbc,Sac,Sabc))
            # print(torch.linalg.eigvals(rho_bc))
            # print(torch.linalg.eigvals(rho_bc).real)
            obs_values = [Wp, Wp2, Mx, My, Mz, XX_z,
                          YY_z, ZZ_z, Qxx, Qyy, Qzz, TEE]

        return obs_values, obs_labels

    def eval_corrf_SS(self, coord, direction, state, env, dist):

        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            # rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            rot_op = torch.eye(
                self.phys_dim, dtype=self.dtype, device=self.device)
            op_0 = op
            op_rot = torch.einsum('ki,kl,lj->ij', rot_op, op_0, rot_op)

            def _gen_op(r):
                # return op_rot if r%2==0 else op_0
                return op_0
            return _gen_op

        op_sx = 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])
        op_isy = -0.5*(self.obs_ops["sp"] - self.obs_ops["sm"])

        Sz0szR = corrf.corrf_1sO1sO(coord, direction, state, env, self.obs_ops["sz"],
                                    conjugate_op(self.obs_ops["sz"]), dist)
        Sx0sxR = corrf.corrf_1sO1sO(
            coord, direction, state, env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR = corrf.corrf_1sO1sO(
            coord, direction, state, env, op_isy, conjugate_op(op_isy), dist)

        res = dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR,
                   "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res

    def gtno(self, params):
        # GTNO
        Q = LG_h(params[0], params[1], params[2])
        G1, G2, cs = G_kitaev_ani_hard(params[3:14])

        A1 = state_111()
        A2 = torch.cos(params[14])*state_m1m1m1() + \
            torch.sin(params[14])*state_111()

        A1Q = torch.einsum('ijklm,jabc->ikalbmc', Q, A1).reshape(2, 2, 2, 2)
        A2Q = torch.einsum('ijklm,jabc->ikalbmc', Q, A2).reshape(2, 2, 2, 2)
        GA1Q = torch.einsum('ijklm,jabc->ikalbmc', G1, A1Q).reshape(2, 4, 4, 4)
        GA2Q = torch.einsum('ijklm,jabc->ikalbmc', G2, A2Q).reshape(2, 4, 4, 4)
        A = torch.einsum('iabc,jamn->ijmcbn', GA1Q,
                         GA2Q).reshape(2**2, 4, 4, 4, 4)
        A = A/torch.max(torch.abs(A))

        def lattice_to_site(coord):
            return (0, 0)
        sites = {(0, 0): A}
        state = IPEPS(sites, vertexToSite=lattice_to_site)

        return state
