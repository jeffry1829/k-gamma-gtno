#python optim_aniso_k.py --CTMARGS_ctm_conv_tol=1e-4 --chi=20 --OPTARGS_tolerance_change=1e-8 --params_out=test --GLOBALARGS_dtype=complex128 --tiling=2SITE --GLOBALARGS_device=cuda:0
# python optim_aniso_k.py --CTMARGS_ctm_conv_tol=1e-2 --chi=20 --OPTARGS_tolerance_change=1e-8 --OPTARGS_momentum=0.9 --OPTARGS_lr=0.5 --params_out=test --GLOBALARGS_dtype=complex128 --tiling=2SITE --GLOBALARGS_device=cuda:0
# python optim_aniso_k_applyLG.py --CTMARGS_ctm_conv_tol=1e-2 --chi=20 --OPTARGS_tolerance_change=1e-8 --OPTARGS_momentum=0.9 --OPTARGS_lr=0.5 --params_out=test --GLOBALARGS_dtype=complex128 --tiling=2SITE --GLOBALARGS_device=cuda:0
# python optim_aniso_k_obs.py --CTMARGS_ctm_conv_tol=1e-3 --OPTARGS_tolerance_change=1e-8 --params_out=test --GLOBALARGS_dtype=complex128 --tiling=2SITE --GLOBALARGS_device=cuda:1

for i in `seq 1.0 0.1 2.0`; do
    echo ${i}
    python EE.py --CTMARGS_ctm_conv_tol=1e-2 --chi=20 --OPTARGS_tolerance_change=1e-8 --OPTARGS_momentum=0.9\
     --OPTARGS_lr=0.5 --params_out=test --GLOBALARGS_dtype=complex128 --GLOBALARGS_device=cuda:0 --Kx=1 --Ky=1 --Kz=${i} --h=0\
     --EEfn=datas/aniksdg_noLG_sweepLG_TS/EE_h0_0.6toCSL_K${i}.txt\
     --topEEfn=datas/aniksdg_noLG_sweepLG_TS/topEE_h0_0.6toCSL_K${i}.txt\
     --tiling=2SITE
done

# python EE.py --CTMARGS_ctm_conv_tol=1e-2 --chi=20 --OPTARGS_tolerance_change=1e-8 --OPTARGS_momentum=0.9 --OPTARGS_lr=0.5 --params_out=test --GLOBALARGS_dtype=complex128 --GLOBALARGS_device=cuda:0 --Kx=1 --Ky=1 --Kz=1.2 --h=0
