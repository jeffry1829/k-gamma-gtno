#!/bin/bash
Jx=-1.0
Jy=-1.0
Jz=-1.0
h=0.0
chi=8
bond_dim=2
_size=11
statefile="KitaevLG.json"
_step=4
_device="cpu"
_dtype="complex128"
datadir="data/Kitaev_Jx${Jx}Jy${Jy}Jz${Jz}h${h}chi${chi}/"
extra_flags="--CTMARGS_ctm_force_dl True"

mkdir -p ${datadir}

rm SW.txt
rm excitedE.txt
rm eigN.txt

# Calculate the energy and spectral weight
# M->X
# for kx in $(seq 24 24); do
for ky in $(seq 0 ${_step} 23); do
    kx=24
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat_Kitaev.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir}
done
# done
# X->S
for kx in $(seq 24 -${_step} 13); do
    # for ky in $(seq 24 -2 13); do
    ky=$kx
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat_Kitaev.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir}
    # done
done
# S->gamma
for kx in $(seq 12 -${_step} 1); do
    ky=$kx
    # for ky in $(seq 12 -2 1); do
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat_Kitaev.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir}
    # done
done
# gamma->M
# for kx in $(seq 12 -2 1); do
for kx in $(seq 0 ${_step} 23); do
    ky=0
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat_Kitaev.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir}
    # done
done
# M->S
# for kx in $(seq 12 -2 1); do
for kx in $(seq 24 -${_step} 12); do
    ky=$((24-$kx))
    echo "kx="${kx} "ky="${ky}
    python SMA_stored_mat_Kitaev.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir}
    # done
done

#Draw the figure
python graph.py ${datadir}
