#!/bin/bash
hx=2.5
chi=16
bond_dim=2
_size=11
statefile="floatD=2TFIM_output_state.json"
_step=4
_device="cuda:0"
_dtype="complex128"
datadir="data/floatTFIM_hx${hx}chi${chi}/"
extra_flags="--CTMARGS_ctm_force_dl True"

mkdir -p ${datadir}

# # M->X
# for ky in $(seq 0 ${_step} 23); do
#     kx=24
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} \
#     --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # X->S
# for kx in $(seq 24 -${_step} 13); do
#     ky=$kx
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} \
#     --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # S->gamma
# for kx in $(seq 12 -${_step} 1); do
#     ky=$kx
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} \
#     --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # gamma->M
# for kx in $(seq 0 ${_step} 23); do
#     ky=0
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} \
#     --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # M->S
# for kx in $(seq 24 -${_step} 12); do
#     ky=$((24-$kx))
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} \
#     --hx ${hx} --datadir ${datadir} ${extra_flags}
# done

# rm ${datadir}SW.txt
# rm ${datadir}excitedE.txt
# rm ${datadir}eigN.txt
# rm ${datadir}TV.txt
# rm ${datadir}SSF.txt

# # M->X
# # for kx in $(seq 24 24); do
# for ky in $(seq 0 ${_step} 23); do
#     kx=24
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA_stored_mat.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # done
# # X->S
# for kx in $(seq 24 -${_step} 13); do
#     ky=$kx
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA_stored_mat.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # S->gamma
# for kx in $(seq 12 -${_step} 1); do
#     ky=$kx
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA_stored_mat.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # gamma->M
# for kx in $(seq 0 ${_step} 23); do
#     ky=0
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA_stored_mat.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
# done
# # M->S
# for kx in $(seq 24 -${_step} 12); do
#     ky=$((24-$kx))
#     echo "kx="${kx} "ky="${ky}
#     python -u SMA_stored_mat.py --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
#     --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 1000 --GLOBALARGS_device ${_device} \
#     --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
# done

#Draw the figure
python -u graph.py ${datadir}