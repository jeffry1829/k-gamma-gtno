python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --kx 1.0 --ky 1.0 --kz 2.1 --h 0 --instate 0.00_state.json --out_prefix 0.00 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 300
a=($(seq 0.00 0.02 0.16))
for i in {0..10}
do
    python ../../optim_kitaev3.py --bond_dim 4 --chi 9 --seed 123 --instate_noise 0.0001 --kx 1.0 --ky 1.0 --kz 2.1 --h ${a[i+1]} --instate ${a[i]}_state.json --out_prefix ${a[i+1]} --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:3 --opt_max_iter 300
done