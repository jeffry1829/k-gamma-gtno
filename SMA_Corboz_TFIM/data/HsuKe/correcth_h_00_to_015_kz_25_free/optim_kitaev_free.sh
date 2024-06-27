# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.02 --instate h_00.json --out_prefix h_002 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.04 --instate h_002_state.json --out_prefix h_004 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.06 --instate h_004_state.json --out_prefix h_006 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.08 --instate h_006_state.json --out_prefix h_008 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.10 --instate h_008_state.json --out_prefix h_010 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.12 --instate h_010_state.json --out_prefix h_012 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.14 --instate h_012_state.json --out_prefix h_014 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2

# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.16 --instate h_014_state.json --out_prefix h_016 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.18 --instate h_016_state.json --out_prefix h_018 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.20 --instate h_018_state.json --out_prefix h_020 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.22 --instate h_020_state.json --out_prefix h_022 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.24 --instate h_022_state.json --out_prefix h_024 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.26 --instate h_024_state.json --out_prefix h_026 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2

# mkdir find050/
# for i in {1..50}
# do
#     python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed ${i} --kx 1.0 --ky 1.0 --kz 2.5 --h 0.50 --ipeps_init_type RANDOM2 --out_prefix find050/h_050-${i} --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# done






# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.50 --instate h_050_state.json --out_prefix h_050 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.50 --instate h_050_state.json --out_prefix h_050 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.48 --instate h_050_state.json --out_prefix h_048 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.46 --instate h_048_state.json --out_prefix h_046 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.44 --instate h_046_state.json --out_prefix h_044 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.42 --instate h_044_state.json --out_prefix h_042 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.40 --instate h_042_state.json --out_prefix h_040 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.38 --instate h_040_state.json --out_prefix h_038 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.36 --instate h_038_state.json --out_prefix h_036 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.34 --instate h_036_state.json --out_prefix h_034 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2




a=($(seq 0 0.02 0.5))
for i in {0..25}
do
    python ../../optim_kitaev3.py --bond_dim 4 --chi 16 --seed 12321 --instate_noise 0.009 --kx 1.0 --ky 1.0\
     --kz 2.5 --h ${a[i+1]} --instate ${a[i]}_state.json --out_prefix ${a[i+1]}\
      --GLOBALARGS_dtype complex128 --GLOBALARGS_device cpu\
      --CTMARGS_projector_eps_multiplet 1e-12 --CTMARGS_projector_svd_reltol 1e-12 --opt_max_iter 100\
       --CTMARGS_ctm_max_iter 100 --CTMARGS_projector_method 4X4\
       --omp_cores 36
done