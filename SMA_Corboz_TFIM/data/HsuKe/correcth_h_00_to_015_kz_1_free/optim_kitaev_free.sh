
# python ../../optim_kitaev3.py --bond_dim 6 --chi 4 --seed 123 --kx 1.0 --ky 1.0 --kz 1.0 --h 0.0 --out_prefix D8_test --GLOBALARGS_dtype complex128 --GLOBALARGS_device cuda:2 --opt_max_iter 3

# a=($(seq 0.0 0.05 1.0))
a=($(seq 0.0 0.05 0.20))
# for i in {0..20}
for i in {0..4}
do
    python ../../optim_kitaev3.py --bond_dim 4 --chi 16 --seed 123 --instate_noise 0.0001 --kx 1.0 --ky 1.0\
     --kz 1.0 --h ${a[i+1]} --instate ${a[i]}_state.json --out_prefix ${a[i+1]}\
      --GLOBALARGS_dtype complex128 --GLOBALARGS_device cpu\
      --CTMARGS_projector_eps_multiplet 1e-12 --CTMARGS_projector_svd_reltol 1e-12 --opt_max_iter 100\
       --CTMARGS_ctm_max_iter 100000 --CTMARGS_projector_method 4X4
done