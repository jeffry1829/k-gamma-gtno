
#python ../../optim_kitaev3.py --bond_dim 2 --chi 20 --seed 123 --kx 0 --ky 0 --kz 0 --h 1 --out_prefix polarized --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1

#python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate polarized_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1

# python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate kitaev_0.5_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
# python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate kitaev_0.5_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
# python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate kitaev_0.5_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
# python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h 0.5 --instate kitaev_0.5_state.json --out_prefix kitaev_0.5 --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1


a=($(seq 0.5 -0.02 0.0))
for i in {0..25}
do
  echo ${a[i]}
  python ../../optim_kitaev3.py --bond_dim 4 --chi 20 --instate_noise 0.001 --seed 123 --kx 1.0 --ky 1.0 --kz 2.5 --h ${a[i+1]} --instate kitaev_${a[i]}_state.json --out_prefix kitaev_${a[i+1]} --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:1
done