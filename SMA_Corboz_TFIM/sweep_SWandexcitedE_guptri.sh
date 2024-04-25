#!/bin/bash
# for hx in $(seq 3.0 0.1 3.1); do
hx=3.0
echo "hx="${hx}
chi=16
bond_dim=4
_size=11
_step=4
_dtype="complex64"
statefile="${_dtype}hx${hx}D${bond_dim}TFIM_output_state.json"
# statefile="hx${hx}D${bond_dim}TFIM_output_state.json"
_device="cuda:0"
datadir="data/h${hx}chi${chi}/"
extra_flags="--CTMARGS_ctm_force_dl True"

mkdir -p ${datadir}

# M->X
# for kx in $(seq 24 24); do
for ky in $(seq 0 ${_step} 23); do
    kx=24
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --datadir ${datadir} ${extra_flags}
done
# done
# X->S
for kx in $(seq 24 -${_step} 13); do
    # for ky in $(seq 24 -2 13); do
    ky=$kx
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --datadir ${datadir} ${extra_flags}
    # done
done
# S->gamma
for kx in $(seq 12 -${_step} 1); do
    ky=$kx
    # for ky in $(seq 12 -2 1); do
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --datadir ${datadir} ${extra_flags}
    # done
done
# gamma->M
# for kx in $(seq 12 -2 1); do
for kx in $(seq 0 ${_step} 23); do
    ky=0
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --datadir ${datadir} ${extra_flags}
    # done
done
# M->S
# for kx in $(seq 12 -2 1); do
for kx in $(seq 24 -${_step} 12); do
    ky=$((24-$kx))
    echo "kx="${kx} "ky="${ky}
    python SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --hx ${hx} --chi ${chi} \
    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
    --kx ${kx} --ky ${ky} --datadir ${datadir} ${extra_flags}
    # done
done

rm ${datadir}guptri_excitedE.txt

# Calculate the energy and spectral weight
# M->X
# for kx in $(seq 24 24); do
for ky in $(seq 0 ${_step} 23); do
    kx=24
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0 ${datadir}
done
# done
# X->S
for kx in $(seq 24 -${_step} 13); do
    # for ky in $(seq 24 -2 13); do
    ky=$kx
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0 ${datadir}
    # done
done
# S->gamma
for kx in $(seq 12 -${_step} 1); do
    ky=$kx
    kx=${kx}.0
    ky=${ky}.0
    # for ky in $(seq 12 -2 1); do
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0 ${datadir}
    # done
done
# gamma->M
# for kx in $(seq 12 -2 1); do
for kx in $(seq 0 ${_step} 23); do
    ky=0
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0 ${datadir}
    # done
done
# M->S
# for kx in $(seq 12 -2 1); do
for kx in $(seq 24 -${_step} 12); do
    ky=$((24-$kx))
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0 ${datadir}
    # done
done

#Draw the figure
python guptri_graph.py ${datadir}
# done