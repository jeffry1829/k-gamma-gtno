#!/bin/bash
for i in `seq 1.0 0.1 2.0`; do
    for j in `seq 0 1 119`; do
        echo "Kz="${i} "h_num="${j}
        python staticstructurefactor_anisok.py --GLOBALARGS_dtype complex128 --bond_dim=4 --chi=8 --size=11 --num_h=${j} --CTMARGS_ctm_conv_tol=1e-3 --OPTARGS_tolerance_change=1e-8 --params_out=test --GLOBALARGS_dtype=complex128 --GLOBALARGS_device=cpu --h=0 --Kx=1 --Ky=1 --Kz=${i} --SSFfn=datas/aniksdg_noLG_sweepLG_SSF/SSF_h0_0.6toCSL_K${i}.txt
    done
done