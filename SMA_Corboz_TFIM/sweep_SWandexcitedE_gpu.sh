#!/bin/bash
hx=2.5
chi=8
bond_dim=2
_size=11
statefile="D=2TFIM_output_state.json"
# _device="cpu"
# M->X
# for kx in $(seq 24 24); do
for ky in $(seq 0 2 23); do
    kx=24
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cuda:0 \
    --kx ${kx} --ky ${ky}
done &
# done
# X->S
for kx in $(seq 24 -2 13); do
    # for ky in $(seq 24 -2 13); do
    ky=$kx
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cuda:1 \
    --kx ${kx} --ky ${ky}
    # done
done &
# S->gamma
for kx in $(seq 12 -2 1); do
    ky=$kx
    # for ky in $(seq 12 -2 1); do
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cuda:2 \
    --kx ${kx} --ky ${ky}
    # done
done &
# gamma->M
# for kx in $(seq 12 -2 1); do
for kx in $(seq 0 2 23); do
    ky=0
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cuda:3 \
    --kx ${kx} --ky ${ky}
    # done
done &

wait

# M->S
# for kx in $(seq 12 -2 1); do
for kx in $(seq 24 -2 12); do
    ky=$((24-$kx))
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device cuda:0 \
    --kx ${kx} --ky ${ky}
    # done
done &

wait

rm SW.txt
rm excitedE.txt
rm eigN.txt

# Calculate the energy and spectral weight
# M->X
# for kx in $(seq 24 24); do
for ky in $(seq 0 2 23); do
    kx=24
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky}
done
# done
# X->S
for kx in $(seq 24 -2 13); do
    # for ky in $(seq 24 -2 13); do
    ky=$kx
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky}
    # done
done
# S->gamma
for kx in $(seq 12 -2 1); do
    ky=$kx
    # for ky in $(seq 12 -2 1); do
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky}
    # done
done
# gamma->M
# for kx in $(seq 12 -2 1); do
for kx in $(seq 0 2 23); do
    ky=0
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky}
    # done
done
# M->S
# for kx in $(seq 12 -2 1); do
for kx in $(seq 24 -2 12); do
    ky=$((24-$kx))
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat.py --GLOBALARGS_dtype complex128 --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky}
    # done
done

#Draw the figure
python graph.py
