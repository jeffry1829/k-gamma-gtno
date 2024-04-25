#!/bin/bash
for i in `seq 1.0 0.1 2.0`; do
    echo ${i}
    python optim_aniso_k_sweepLG2.py --CTMARGS_ctm_conv_tol=1e-3 --OPTARGS_tolerance_change=1e-8 --params_out=test --GLOBALARGS_dtype=complex128 --tiling=2SITE --GLOBALARGS_device=cuda:3 --h=0 --Kx=1 --Ky=1 --Kz=${i}\
    --Efn=datas/anikaniksgd_sweepsgd/E_h0_0.6toCSL_K${i}.txt --magfn=datas/aniksgd_sweep/mag_h0_0.6toCSL_K${i}.txt --csfn=datas/aniksgd_sweep/cs_h0_0.6toCSL_K${i}.txt\
    --Wfn=datas/aniksgd_sweep/W_h0_0.6toCSL_K${i}.txt
done