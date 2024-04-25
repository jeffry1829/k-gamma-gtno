# python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.0 --ipeps_init_type LG --p 1 --out_prefix h_00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3

python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.0 --instate h_00_state.json --out_prefix h_00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250

python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.02 --instate h_00_state.json --out_prefix h_002 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.04 --instate h_002_state.json --out_prefix h_004 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.06 --instate h_004_state.json --out_prefix h_006 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.08 --instate h_006_state.json --out_prefix h_008 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.10 --instate h_008_state.json --out_prefix h_010 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.12 --instate h_010_state.json --out_prefix h_012 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.14 --instate h_012_state.json --out_prefix h_014 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 250