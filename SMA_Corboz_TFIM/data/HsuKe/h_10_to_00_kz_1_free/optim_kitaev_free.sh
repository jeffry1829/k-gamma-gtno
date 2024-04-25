
# python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 1.0 --h 1.0 --instate polarized_state.json --out_prefix 1.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 1.0 --h 1.0 --instate 1.00_state.json --out_prefix 1.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 1.0 --h 1.0 --instate 1.00_state.json --out_prefix 1.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 1.0 --h 1.0 --instate 1.00_state.json --out_prefix 1.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2


a=($(seq 1.0 -0.05 0.0))
for i in {0..20}
do
    python ../../optim_kitaev3.py --bond_dim 4 --chi 4 --seed 123 --instate_noise 0.0001 --kx 1.0 --ky 1.0 --kz 1.0 --h ${a[i+1]} --instate ${a[i]}_state.json --out_prefix ${a[i+1]} --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2
done