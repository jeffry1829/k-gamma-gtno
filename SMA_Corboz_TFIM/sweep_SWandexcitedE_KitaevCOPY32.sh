#!/bin/bash
Jx=-1.0
Jy=-1.0
Jz=-1.0
h=0.0
chi=32
bond_dim=2
_size=11
statefile="KitaevLG.json"
_step=4
_device="cuda:1"
_dtype="complex128"
datadir="data/Kitaev_Jx${Jx}Jy${Jy}Jz${Jz}h${h}chi${chi}size${_size}bonddim${bond_dim}dtype${_dtype}/"
reuseCTMRGenv="True"
removeCTMRGenv="False"
extra_flags="--CTMARGS_ctm_force_dl True --MultiGPU True --CTMARGS_projector_eps_multiplet 1e-4 --CTMARGS_ctm_conv_tol 1e-8"
# --CTMARGS_fpcm_freq 3 --CTMARGS_fpcm_fpt_tol 2e-6 --CTMARGS_fpcm_isogauge_tol 2e-6"
StoredMatMethod="SMA_stored_mat_Kitaev.py"
SMAMethod="SMA_Kitaev.py"
runSMA="False"
runStoredMat="False"
runDraw="True"
runGUPTRI="False"
runDrawGUPTRI="False"

mkdir -p ${datadir}

if [[ "$runSMA" == "True" ]]; then
    # Correct Path:
    # M->Gamma
    for ky in $(seq 24 -${_step} 1); do
        kx=0
        echo "kx="${kx} "ky="${ky}
        python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
        --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 10000 --GLOBALARGS_device ${_device} \
        --kx ${kx} --ky ${ky} \
        --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
        --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
    done
    # Gamma->K
    for ky in $(seq 0 ${_step} 23); do
        kx=$(($ky+$ky))
        echo "kx="${kx} "ky="${ky}
        python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
        --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 10000 --GLOBALARGS_device ${_device} \
        --kx ${kx} --ky ${ky} \
        --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
        --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
    done
    # K->M
    for kx in $(seq 48 -${_step} 0); do
        ky=24
        # for ky in $(seq 12 -2 1); do
        echo "kx="${kx} "ky="${ky}
        python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
        --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 10000 --GLOBALARGS_device ${_device} \
        --kx ${kx} --ky ${ky} \
        --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
        --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
        # done
    done
fi

if [[ "$runStoredMat" == "True" ]]; then
    rm ${datadir}SW.txt
    rm ${datadir}excitedE.txt
    rm ${datadir}eigN.txt
    rm ${datadir}TV.txt
    rm ${datadir}SSF.txt

    # Calculate the energy and spectral weight

    # Correct Path:
    # M->Gamma
    for ky in $(seq 24 -${_step} 1); do
        kx=0
        echo "kx="${kx} "ky="${ky}
        python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
        --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 10000 --GLOBALARGS_device ${_device} \
        --kx ${kx} --ky ${ky} \
        --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
        --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
    done
    # Gamma->K
    for ky in $(seq 0 ${_step} 23); do
        kx=$(($ky+$ky))
        echo "kx="${kx} "ky="${ky}
        python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
        --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 10000 --GLOBALARGS_device ${_device} \
        --kx ${kx} --ky ${ky} \
        --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
        --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
    done
    # K->M
    for kx in $(seq 48 -${_step} 0); do
        ky=24
        echo "kx="${kx} "ky="${ky}
        python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
        --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter 10000 --GLOBALARGS_device ${_device} \
        --kx ${kx} --ky ${ky} \
        --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
        --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
    done
fi

if [[ "$runDraw" == "True" ]]; then
    #Draw the figure
    python -u graph.py ${datadir}
fi







if [[ "$runGUPTRI" == "True" ]]; then
    # GUPTRI
    rm ${datadir}guptri_excitedE.txt


    # Correct Path:
    # M->Gamma
    for ky in $(seq 24 -${_step} 1); do
        kx=0
        kx=${kx}.0
        ky=${ky}.0
        echo "kx="${kx} "ky="${ky}
        python -u guptri_Kitaev.py ${kx} ${ky} ${Jx} ${Jy} ${Jz} ${h} ${datadir}
    done
    # Gamma->K
    for ky in $(seq 0 ${_step} 23); do
        kx=$(($ky+$ky))
        kx=${kx}.0
        ky=${ky}.0
        echo "kx="${kx} "ky="${ky}
        python -u guptri_Kitaev.py ${kx} ${ky} ${Jx} ${Jy} ${Jz} ${h} ${datadir}
    done
    # K->M
    for kx in $(seq 48 -${_step} 0); do
        ky=24
        kx=${kx}.0
        ky=${ky}.0
        echo "kx="${kx} "ky="${ky}
        python -u guptri_Kitaev.py ${kx} ${ky} ${Jx} ${Jy} ${Jz} ${h} ${datadir}
    done
fi

if [[ "$runDrawGUPTRI" == "True" ]]; then
    #Draw the figure
    python -u guptri_graph.py ${datadir}
fi