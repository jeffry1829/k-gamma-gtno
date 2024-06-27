#!/bin/bash
Kz=1.0
for h in 0.10
# for h in 0.00 0.05 0.10 0.15
do
    Jx=1.0
    Jy=1.0
    Jz=${Kz}
    # h=0.0
    chi=16
    bond_dim=4
    _size=2
    L=$((2*${_size}+2))
    tL=$((2*${L}))
    Lm1=$((L-1))
    hL=$((L/2))
    hLp1=$((hL+1))
    _step=3
    _device="cuda:2"
    _dtype="complex128"
    statefile="${h}_state.json"
    # datadir="data/HsuKe/h_00_to_015_kz_1_free/KitaevAnyon_withP_Jx${Jx}Jy${Jy}Jz${Jz}h${h}chi${chi}size${_size}bonddim${bond_dim}dtype${_dtype}/"
    datadir="data/HsuKe/correcth_h_00_to_015_kz_1_free/KitaevAnyon_Pderv_Jx${Jx}Jy${Jy}Jz${Jz}h${h}chi${chi}size${_size}bonddim${bond_dim}dtype${_dtype}/"
    mkdir -p ${datadir}
    cp ${datadir}../${statefile} ${datadir}
    reuseCTMRGenv="True"
    removeCTMRGenv="False"
    extra_flags="--CTMARGS_ctm_force_dl True --MultiGPU False --CTMARGS_projector_eps_multiplet 1e-12 --CTMARGS_ctm_conv_tol 1e-7"
    extra_flags=${extra_flags}" --NormMat True --HamiMat True --CTMARGS_projector_svd_reltol 1e-12"
    extra_flags=${extra_flags}" --CTMARGS_projector_method 4X4 --CTMARGS_projector_svd_method GESDD_CPU"
    extra_flags=${extra_flags}" --CTMARGS_ctm_env_init_type CTMRG --UseVUMPSansazAC False"
    extra_flags=${extra_flags}" --CTMARGS_ctm_absorb_normalization inf"
    SMAMethod="SMA_Kitaev_Correct_Model_gpugraph_divide_Pderv.py"
    StoredMatMethod="SMA_stored_mat_Kitaev_Correct_Model.py"
    runSMA="F"
    runStoredMat="True"
    runDraw="True"
    runGUPTRI="False"
    runDrawGUPTRI="False"
    skipCalculatedPoints="True"

    OnlyOnePoint="False"
    max_iter=100000000

    # mkdir -p ${datadir}

    if [[ "$runSMA" == "True" ]]; then
        if [[ "$OnlyOnePoint" == "True" ]]; then
            kx=${L}
            ky=0
            echo "kx="${kx} "ky="${ky}
            python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
            --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
            --kx ${kx} --ky ${ky} \
            --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
            --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
        else
            # Correct Path:
            # M->Gamma
            for kx in $(seq ${hL} -${_step} 1); do
                ky=${kx}
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
            done
            # Gamma->K
            for ky in $(seq 0 ${_step} ${Lm1}); do
                kx=0
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
            done
            # K->M
            for ky in $(seq ${L} -${_step} ${hL}); do
                kx=$((${L}-$ky))
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                    # done
                fi
            done
        fi
    fi

    if [[ "$runStoredMat" == "True" ]]; then
        # Check if the user really wants to remove the data
        echo "Are you sure you want to remove the data? (y/n) (input within 10 secs)"
        read -t 10 answer
        if [ $? != 0 ]; then
            echo "You're not inputing anything! default to 'y'"
            answer="y"
        fi
        if [ "$answer" == "y" ]; then
            rm ${datadir}SW.txt
            rm ${datadir}excitedE.txt
            rm ${datadir}eigN.txt
            rm ${datadir}TV.txt
            rm ${datadir}SSF.txt
            rm ${datadir}XXA.txt
            rm ${datadir}XIIXA.txt
            echo "Data removed"


            # Calculate the energy and spectral weight
            if [[ "$OnlyOnePoint" == "True" ]]; then
                kx=${L}
                ky=0
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            else
                # Correct Path:
                # M->Gamma
                for kx in $(seq ${hL} -${_step} 1); do
                    ky=${kx}
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                done
                # Gamma->K
                for ky in $(seq 0 ${_step} ${Lm1}); do
                    kx=0
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                done
                # K->M
                for ky in $(seq ${L} -${_step} ${hL}); do
                    kx=$((${L}-$ky))
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --Jx ${Jx} --Jy ${Jy} --Jz ${Jz} --h ${h} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                done
            fi
        else
            echo "Data not removed and not running"
        fi
    fi

    if [[ "$runDraw" == "True" ]]; then
        #Draw the figure
        python -u graph.py ${datadir} True "h=${h} Kz=${Kz}" "[0, 1, 2, 4], [r'\$M(\pi,0)\$', r'\$\Gamma(0,0)\$',r'\$K(\pi,\frac{\pi}{2})\$', r'\$M(\pi,0)\$']"
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
done