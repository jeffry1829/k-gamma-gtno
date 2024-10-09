import torch
import groups.su2 as su2
from ctm.generic import rdm
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import corrf_c4v
import config as cfg
from math import sqrt
import itertools
import numpy as np


def _cast_to_real(t):
    return t.real if t.is_complex() else t


'''
Current setup:
    - 2x2 unit cell
'''


class ANISO_K():
    def __init__(self, Kx, Ky, Kz, h, global_args=cfg.global_args):

        self.tmp_rdm = None
        self.tmp_Es = None

        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = 2
        # self.zz, self.xx, self.yy, self.hp = self.get_h_2x2()
        self.Kx, self.Ky, self.Kz, self.h = Kx, Ky, Kz, h
        self.x_inter_K, self.y_inter_K, self.z_inter_K, self.local_h = self.get_h_2x2()
        self.obs_ops = self.get_obs_ops()

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

    def get_h_2x2(self):
        # Note that these should act on a 2x2 rdm
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Sz = 2*s2.SZ()
        Sx = s2.SP()+s2.SM()
        Sy = -(s2.SP()-s2.SM())*1j
        XX = torch.einsum('ij,ab->iajb', Sz, Sz)
        ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sx, Sx, Id)
        YYII = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sy, Sy, Id)
        H = 0.5*(torch.einsum('ij,ab->iajb', Sx+Sy+Sz, Id) +
                 torch.einsum('ij,ab->iajb', Id, Sx+Sy+Sz))
        # H = torch.einsum('ij,ab->iajb',Sx+Sy+Sz,Id)# + torch.einsum('ij,ab->iajb',Id, Sx+Sy+Sz)
        return XX, YYII, ZZII, H

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"] = 2*s2.SZ()
        obs_ops["sp"] = 2*s2.SP()
        obs_ops["sm"] = 2*s2.SM()
        return obs_ops

    def energy_2x2(self, state, env, kx, ky, kz, h):
        tmp_rdm = rdm.rdm2x2((0,0), state, env)
        self.tmp_rdm = tmp_rdm
        self.state = state
        ExK = kx*torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype),
                                self.x_inter_K.reshape(4, 4).type(self.dtype))
        energy_per_site = ExK
        EyK = ky*torch.einsum('iajbmanb,ijmn', tmp_rdm.type(self.dtype),
                                self.y_inter_K.reshape(4, 4, 4, 4).type(self.dtype))
        energy_per_site += EyK
        EzK = kz*torch.einsum('ijabmnab,ijmn', tmp_rdm.type(self.dtype),
                                self.z_inter_K.reshape(4, 4, 4, 4).type(self.dtype))
        energy_per_site += EzK
        energy_per_site += h*torch.einsum('iabcjabc,ji', tmp_rdm.type(
            self.dtype), self.local_h.reshape(4, 4).type(self.dtype))*4

        energy_per_site = _cast_to_real(energy_per_site)
        energy_per_site *= (1/8)

        return energy_per_site
    
    def energy_2x2_2site(self, state, env, kx, ky, kz, h):
        energy_per_site = torch.zeros(1, dtype=self.dtype, device=self.device)
        Es = []
        ExK = EyK = EzK = 0
        for coord in state.sites.keys():
            Es = []
            # print("coordinate:")
            # print(coord)
            tmp_rdm = rdm.rdm2x2(coord, state, env)
            self.tmp_rdm = tmp_rdm
            self.state = state
            ExK = kx*torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype),
                                  self.x_inter_K.reshape(4, 4).type(self.dtype))
            energy_per_site += ExK
            EyK = ky*torch.einsum('iajbmanb,mnij', tmp_rdm.type(self.dtype),
                                  self.y_inter_K.reshape(4, 4, 4, 4).type(self.dtype))
            energy_per_site += EyK
            EzK = kz*torch.einsum('ijabmnab,mnij', tmp_rdm.type(self.dtype),
                                  self.z_inter_K.reshape(4, 4, 4, 4).type(self.dtype))
            energy_per_site += EzK
            energy_per_site += h*torch.einsum('iabcjabc,ji', tmp_rdm.type(
                self.dtype), self.local_h.reshape(4, 4).type(self.dtype))*4

        energy_per_site = _cast_to_real(energy_per_site)
        energy_per_site *= (1/16)

        Es.extend([(1/16)*ExK, (1/16)*EyK, (1/16)*EzK, energy_per_site])
        self.tmp_Es = Es

        return energy_per_site

    def eval_obs(self, state, env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. :math:`\langle 2S^z \rangle,\ \langle 2S^x \rangle` for each site in the unit cell

        """
        obs = dict()
        with torch.no_grad():
            for coord, site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord, state, env)
                for label, op in self.obs_ops.items():
                    obs[f"{label}{coord}"] = torch.trace(rdm1x1@op)
                obs[f"sx{coord}"] = 0.5*(obs[f"sp{coord}"] + obs[f"sm{coord}"])

            for coord, site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord, state, env)
                rdm1x2 = rdm.rdm1x2(coord, state, env)
                rdm2x2 = rdm.rdm2x2(coord, state, env)
                SzSz2x1 = torch.einsum('ijab,ijab', rdm2x1, self.h2)
                SzSz1x2 = torch.einsum('ijab,ijab', rdm1x2, self.h2)
                SzSzSzSz = torch.einsum('ijklabcd,ijklabcd', rdm2x2, self.h4)
                obs[f"SzSz2x1{coord}"] = _cast_to_real(SzSz2x1)
                obs[f"SzSz1x2{coord}"] = _cast_to_real(SzSz1x2)
                obs[f"SzSzSzSz{coord}"] = _cast_to_real(SzSzSzSz)

        # prepare list with labels and values
        obs_labels = [f"{lc[1]}{lc[0]}" for lc in list(
            itertools.product(state.sites.keys(), ["sz", "sx"]))]
        obs_labels += [f"SzSz2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SzSz1x2{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SzSzSzSz{coord}" for coord in state.sites.keys()]
        obs_values = [obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def get_XX(self, state, env):
        # s0 s1
        # s2 s3
        # Note that by wraping with tensor will result in req_grad related errors!
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Z = 2*s2.SZ()
        X = s2.SP()+s2.SM()
        Y = -(s2.SP()-s2.SM())*1j

        IX = torch.einsum('ij,ab->iajb', Id, X).reshape(4, 4).type(self.dtype)
        XI = torch.einsum('ij,ab->iajb', X, Id).reshape(4, 4).type(self.dtype)

        II = torch.einsum('ij,ab->iajb', Id, Id).reshape(4, 4).type(self.dtype)
        # s0 s1
        # s2 s3
        rdm2x2 = rdm.rdm2x2((0, 0), state, env)  # s0,s1,s2,s3,s0',s1',s2',s3'
        XX = torch.einsum('abcdlmno,la,mb,nc,od',
                          rdm2x2.type(self.dtype), II, XI, II, II)
        XX = _cast_to_real(XX)
        return XX

    def get_ZZ(self, state, env):
        # s0 s1
        # s2 s3
        # Note that by wraping with tensor will result in req_grad related errors!
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Z = 2*s2.SZ()
        X = s2.SP()+s2.SM()
        Y = -(s2.SP()-s2.SM())*1j

        IZ = torch.einsum('ij,ab->iajb', Id, Z).reshape(4, 4).type(self.dtype)
        ZI = torch.einsum('ij,ab->iajb', Z, Id).reshape(4, 4).type(self.dtype)

        II = torch.einsum('ij,ab->iajb', Id, Id).reshape(4, 4).type(self.dtype)
        # s0 s1
        # s2 s3
        rdm2x2 = rdm.rdm2x2((0, 0), state, env)  # s0,s1,s2,s3,s0',s1',s2',s3'
        ZZ = torch.einsum('abcdlmno,la,mb,nc,od',
                          rdm2x2.type(self.dtype), II, ZI, II, II)
        ZZ = _cast_to_real(ZZ)
        return ZZ

    def get_YY(self, state, env):
        # s0 s1
        # s2 s3
        # Note that by wraping with tensor will result in req_grad related errors!
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Z = 2*s2.SZ()
        X = s2.SP()+s2.SM()
        Y = -(s2.SP()-s2.SM())*1j

        IZ = torch.einsum('ij,ab->iajb', Id, Y).reshape(4, 4).type(self.dtype)
        ZI = torch.einsum('ij,ab->iajb', Y, Id).reshape(4, 4).type(self.dtype)

        II = torch.einsum('ij,ab->iajb', Id, Id).reshape(4, 4).type(self.dtype)
        # s0 s1
        # s2 s3
        rdm2x2 = rdm.rdm2x2((0, 0), state, env)  # s0,s1,s2,s3,s0',s1',s2',s3'
        ZZ = torch.einsum('abcdlmno,la,mb,nc,od',
                          rdm2x2.type(self.dtype), II, ZI, II, II)
        ZZ = _cast_to_real(ZZ)
        return ZZ

    @torch.no_grad()
    def get_m(self):
        tmp_rdm = self.tmp_rdm
        mag_A1 = []  # [mx, my, mz]
        mag_A2 = []
        mag_B1 = []
        mag_B2 = []
        mag_A1.append(torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), self.XI.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_A1.append(torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), self.YI.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_A1.append(torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), self.ZI.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_A2.append(torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), self.IX.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_A2.append(torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), self.IY.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_A2.append(torch.einsum('iabcjabc,ji', tmp_rdm.type(self.dtype), self.IZ.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_B1.append(torch.einsum('aibcajbc,ji', tmp_rdm.type(self.dtype), self.XI.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_B1.append(torch.einsum('aibcajbc,ji', tmp_rdm.type(self.dtype), self.YI.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_B1.append(torch.einsum('aibcajbc,ji', tmp_rdm.type(self.dtype), self.ZI.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_B2.append(torch.einsum('aibcajbc,ji', tmp_rdm.type(self.dtype), self.IX.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_B2.append(torch.einsum('aibcajbc,ji', tmp_rdm.type(self.dtype), self.IY.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        mag_B2.append(torch.einsum('aibcajbc,ji', tmp_rdm.type(self.dtype), self.IZ.reshape(
            4, 4).type(self.dtype)).detach().clone().cpu().real.item())
        # print("M_A1: {}\nM_A2: {}\nM_B1: {}\nM_B2: {}".format(mag_A1,mag_A2,mag_B1,mag_B2))
        return mag_A1+mag_A2+mag_B1+mag_B2

    @torch.no_grad()
    def get_Qxx(self):
        tmp_rdm = self.tmp_rdm
        state = self.state
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Sz = 2*s2.SZ()
        Sx = s2.SP()+s2.SM()
        Sy = -(s2.SP()-s2.SM())*1j
        XX = torch.einsum('ij,ab,cd,ef->iacejbdf', Id, Sx, Sx,
                          Id).reshape(4, 4, 4, 4).type(self.dtype)
        # YY = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sy, Sy, Id).reshape(4,4,4,4).type(self.dtype)
        # ZZ = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sz, Sz, Id).reshape(4,4,4,4).type(self.dtype)
        Qxx = torch.zeros(1, dtype=self.dtype, device=self.device)
        for coord in state.sites.keys():
            Qxx += torch.einsum('ijabmnab,mnij', tmp_rdm.type(self.dtype), XX)
        Qxx = _cast_to_real(Qxx)
        Qxx *= (1/16)
        return Qxx.clone().cpu().real.item()

    @torch.no_grad()
    def get_E(self):
        Es = []
        for ele in self.tmp_Es:
            Es.append(ele.detach().clone().cpu().real.item())
        return Es

    # @torch.no_grad()
    # def get_W(self):
    #     tmp_rdm= self.tmp_rdm
    #     state = self.state
    #     s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
    #     Id= torch.eye(2,dtype=self.dtype,device=self.device)
    #     Sz = 2*s2.SZ()
    #     Sx = s2.SP()+s2.SM()
    #     Sy = -(s2.SP()-s2.SM())*1j
    #     XX = torch.einsum('ij,ab->iajb',Sx,Sx)
    #     ZZII = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sz, Sz, Id)
    #     YYII = torch.einsum('ij,ab,cd,ef->iacejbdf',Id, Sy, Sy, Id)
    #     YZ = torch.einsum('ij,ab->iajb',Sy, Sz).reshape(4,4)
    #     ZY = torch.einsum('ij,ab->iajb',Sz, Sy).reshape(4,4)
    #     XI = torch.einsum('ij,ab->iajb',Sx, Id).reshape(4,4)
    #     IX = torch.einsum('ij,ab->iajb',Id, Sx).reshape(4,4)

    #     W = torch.einsum('ijklabcd,ai,bj,ck,dl', tmp_rdm.type(self.dtype), XI, YZ, ZY, IX)
    #     return W.detach().clone().cpu().real.item()

    @torch.no_grad()
    def get_Wp(self):
        tmp_rdm = self.tmp_rdm
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Z = 2*s2.SZ()
        X = s2.SP()+s2.SM()
        Y = -(s2.SP()-s2.SM())*1j
        # IZ = torch.einsum('ij,ab->iajb',Id,Z).reshape(4,4).type(self.dtype)
        # ZI = torch.einsum('ij,ab->iajb',Z,Id).reshape(4,4).type(self.dtype)
        # XY = torch.einsum('ij,ab->iajb',X,Y).reshape(4,4).type(self.dtype)
        # YX = torch.einsum('ij,ab->iajb',Y,X).reshape(4,4).type(self.dtype)
        XI = torch.einsum('ij,ab->iajb', Id, X).reshape(4, 4).type(self.dtype)
        IX = torch.einsum('ij,ab->iajb', X, Id).reshape(4, 4).type(self.dtype)
        YZ = torch.einsum('ij,ab->iajb', Y, Z).reshape(4, 4).type(self.dtype)
        ZY = torch.einsum('ij,ab->iajb', Z, Y).reshape(4, 4).type(self.dtype)
        Wp = torch.einsum('abcdlmno,la,mb,nc,od',
                          tmp_rdm.type(self.dtype), XI, YZ, ZY, IX)
        # Wp = torch.einsum('abcdabcd',rdm2x2.type(self.dtype))
        Wp = _cast_to_real(Wp)
        print("Wp = ", Wp)
        return Wp.detach().clone().cpu().real.item()

    @torch.no_grad()
    def get_TotalSpin(self):
        tmp_rdm = self.tmp_rdm
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        Id = torch.eye(2, dtype=self.dtype, device=self.device)
        Z = 2*s2.SZ()
        X = s2.SP()+s2.SM()
        Y = -(s2.SP()-s2.SM())*1j
        XI = torch.einsum('ij,ab->iajb', X, Id).reshape(4, 4).type(self.dtype)
        IX = torch.einsum('ij,ab->iajb', Id, X).reshape(4, 4).type(self.dtype)
        YI = torch.einsum('ij,ab->iajb', Y, Id).reshape(4, 4).type(self.dtype)
        IY = torch.einsum('ij,ab->iajb', Id, Y).reshape(4, 4).type(self.dtype)
        ZI = torch.einsum('ij,ab->iajb', Z, Id).reshape(4, 4).type(self.dtype)
        IZ = torch.einsum('ij,ab->iajb', Id, Z).reshape(4, 4).type(self.dtype)

        ops = [IX+XI, IY+YI, IZ+ZI]

        state = self.state
        TotalSpin = 0
        Len = 1
        phy = 4
        vir = 4

        def sn(i, j):  # site number
            return (i+Len)*2*Len+(j+Len)+1

        # assume permutation:
        #      0
        #    1 T 3
        #      2
        #      |----------> j
        #      |
        #      |
        #      |
        #      v
        #      i
        peps = state.site((-Len, -Len)).clone()[:, 0, 0, :, :]
        for i in range(-Len, Len):
            for j in range(-Len, Len):
                if i == j == -Len:
                    continue
                if i == -Len and j != -Len:
                    if j != Len-1:
                        peps = torch.einsum(
                            'iab,cbgh->icagh', peps, state.site((i, j))[:, 0, :, :, :])\
                            .reshape(phy**(sn(i, j)), vir**(Len+j+1), vir)
                    else:
                        peps = torch.einsum(
                            'iab,cbg->icag', peps, state.site((i, j))[:, 0, :, :, 0])\
                            .reshape(phy**(sn(i, j)), vir**(2*Len))
                elif i != -Len and j == -Len:
                    if i != Len-1:
                        peps = peps.reshape(
                            phy**(sn(i, j)-1), vir, vir**(2*Len-1))
                        peps = torch.einsum(
                            'abi,cbgh->acigh', peps, state.site((i, j))[:, :, 0, :, :])\
                            .reshape(phy**(sn(i, j)), vir**(2*Len-1), vir, vir)
                    else:
                        peps = peps.reshape(
                            phy**(sn(i, j)-1), vir, vir**(2*Len-1))
                        peps = torch.einsum(
                            'abi,cbh->acih', peps, state.site((i, j))[:, :, 0, 0, :])\
                            .reshape(phy**(sn(i, j)), vir**(2*Len-1), vir)
                elif i == Len-1:
                    if j != Len-1:
                        peps = peps.reshape(phy**(sn(i, j)-1),
                                            vir, vir**(Len-1-j), vir)
                        peps = torch.einsum('abcd,ebdj->aecj', peps, state.site((i, j))[:, :, :, 0, :])\
                            .reshape(phy**(sn(i, j)), vir**(Len-1-j), vir)
                    else:
                        peps = peps.reshape(phy**(sn(i, j)-1),
                                            vir, vir)
                        peps = torch.einsum('abc,dbc->ad', peps, state.site((i, j))[:, :, :, 0, 0])\
                            .reshape(phy**(sn(i, j)))
                elif j == Len-1:
                    peps = peps.reshape(phy**(sn(i, j)-1),
                                        vir, vir**(2*Len-1), vir)
                    peps = torch.einsum('abcd,ebdi->aeci', peps, state.site((i, j))[:, :, :, :, 0])\
                        .reshape(phy**(sn(i, j)), vir**(2*Len))
                elif i != -Len and j != -Len and i != Len-1 and j != Len-1:
                    peps = peps.reshape(phy**(sn(i, j)-1),
                                        vir, vir**(2*Len-1), vir)
                    peps = torch.einsum('abce,fbejk->afcjk', peps, state.site((i, j))[:, :, :, :, :])\
                        .reshape(phy**(sn(i, j)), vir**(2*Len), vir)
                else:
                    print('wtf')
        print(peps.size())

        sitenum = (2*Len)**2
        # splitarr = [phy for i in range(2*sitenum)]
        # peps = torch.reshape(peps, tuple(splitarr))
        # permutearr = []
        # for i in range(sitenum):
        #     permutearr.append(i)
        #     permutearr.append(sitenum+i)
        # peps = torch.permute(peps, tuple(permutearr))

        TotalSpin = torch.zeros(1, dtype=self.dtype, device=self.device)
        for i in range(-Len, Len):
            for j in range(-Len, Len):
                for k in range(-Len, Len):
                    for l in range(-Len, Len):
                        v1 = (i, j)
                        v2 = (k, l)
                        if (v1 == v2):
                            continue
                        else:
                            if (sn(i, j) > sn(k, l)):
                                v1, v2 = v2, v1
                            peps = peps.reshape(
                                phy**(sn(v1[0], v1[1])-1), phy, phy**(sn(v2[0],
                                                                         v2[1])-sn(v1[0], v1[1])-1),
                                phy, phy**(sitenum-sn(v2[0], v2[1])))
                            peps2 = peps.clone()
                            for s in range(3):
                                op = ops[s]
                                TotalSpin += torch.einsum(
                                    'abcde,ib,jd,aicje', peps, op, op, peps2.conj())
        print("original TotalSpin = ", TotalSpin)
        TotalSpin = _cast_to_real(TotalSpin)
        return TotalSpin.detach().clone().cpu().real.item()
