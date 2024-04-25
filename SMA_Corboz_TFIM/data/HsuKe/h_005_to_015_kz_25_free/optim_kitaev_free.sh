
# python ../../optim_kitaev3.py --bond_dim 2 --chi 4 --seed 123 --kx 0 --ky 0 --kz 1.0 --h 0.0 --ipeps_init_type RANDOM2 --out_prefix 0.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1

# python ../../optim_kitaev3.py --bond_dim 2 --chi 9 --seed 123 --kx 0 --ky 0 --kz 1.0 --h 0.0 --instate 0.00_state.json --out_prefix 0.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
# python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate kitaev_0.5_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
# python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate kitaev_0.5_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.05 --instate 0.00_state.json --out_prefix 0.050 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3
python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.05 --instate 0.050_state.json --out_prefix 0.050 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3
python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.05 --instate 0.050_state.json --out_prefix 0.050 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3
python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.05 --instate 0.050_state.json --out_prefix 0.050 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3

a=($(seq 0.05 0.005 0.15))
for i in {0..20}
do
  python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --instate_noise 0.001 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h ${a[i+1]} --instate ${a[i]}_state.json --out_prefix ${a[i+1]} --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda
done