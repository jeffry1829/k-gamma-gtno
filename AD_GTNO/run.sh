#!/bin/bash
for i in `seq 1.0 0.1 2.0`; do
    echo ${i}
    python optim_aniso_k.py --CTMARGS_ctm_conv_tol=1e-3 --OPTARGS_tolerance_change=1e-8 --params_out=test --GLOBALARGS_dtype=complex128 --tiling=2SITE --GLOBALARGS_device=cuda:3 --h=0 --Kx=1 --Ky=1 --Kz=${i}\
    --Efn=datas/aniksgdvarLG/E_h0_0.6toCSL_K${i}.txt --magfn=datas/aniksgdvarLG/mag_h0_0.6toCSL_K${i}.txt --csfn=datas/aniksgdvarLG/cs_h0_0.6toCSL_K${i}.txt\
    --Wfn=datas/aniksgdvarLG/W_h0_0.6toCSL_K${i}.txt
done