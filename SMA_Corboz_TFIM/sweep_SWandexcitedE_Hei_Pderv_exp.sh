#!/bin/bash
for j2 in 0.0
do
    j1=1.0
    # j2=${j2}
    chi=16
    bond_dim=2
    _size=3
    L=$((2*${_size}+2))
    Lm1=$((L-1))
    hL=$((L/2))
    hLp1=$((hL+1))
    _step=4
    _device="cuda:1"
    _dtype="complex128"
    statefile="ex-j20D2chi${chi}c4vdtypefloat64_state.json"
    datadir="data/Hei_Pderv_j1${j1}j2${j2}chi${chi}size${_size}bonddim${bond_dim}dtype${_dtype}/"
    mkdir -p ${datadir}
    cp ${datadir}../${statefile} ${datadir}
    reuseCTMRGenv="True"
    removeCTMRGenv="False"
    extra_flags="--CTMARGS_ctm_force_dl True --MultiGPU False --CTMARGS_projector_eps_multiplet 1e-4 --CTMARGS_ctm_conv_tol 1e-8"
    extra_flags=${extra_flags}" --NormMat True --HamiMat True --CTMARGS_projector_svd_reltol 1e-8"
    extra_flags=${extra_flags}" --CTMARGS_projector_method 4X4 --CTMARGS_projector_svd_method GESDD_CPU"
    extra_flags=${extra_flags}" --CTMARGS_ctm_env_init_type CTMRG --UseVUMPSansazAC False"
    extra_flags=${extra_flags}" --CTMARGS_ctm_absorb_normalization inf"
    SMAMethod="SMA_Hei_gpugraph_divide_Pderv.py"
    StoredMatMethod="SMA_stored_mat_Hei.py"
    runSMA="True"
    runStoredMat="True"
    runDraw="True"
    runGUPTRI="False"
    runDrawGUPTRI="False"

    OnlyOnePoint="True"
    max_iter=100000000

    # mkdir -p ${datadir}

    if [[ "$runSMA" == "True" ]]; then
        if [[ "$OnlyOnePoint" == "True" ]]; then
            kx=8
            ky=0
            echo "kx="${kx} "ky="${ky}
            python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
            --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
            --kx ${kx} --ky ${ky} \
            --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
            --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
        else
            # Correct Path:
            # M->X (pi,0)->(pi,pi)
            for ky in $(seq 0 ${_step} ${Lm1}); do
                kx=${L}
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
            # X->S (pi,pi)->(pi/2,pi/2)
            for kx in $(seq ${L} -${_step} ${hLp1}); do
                ky=${kx}
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
            # S->Gamma (pi/2,pi/2)->(0,0)
            for kx in $(seq ${hL} -${_step} 1); do
                ky=${kx}
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                # done
            done
            # Gamma->M (0,0)->(pi,0)
            for kx in $(seq 0 ${_step} ${Lm1}); do
                ky=0
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                # done
            done
            # M->S (pi,0)->(pi/2,pi/2)
            for kx in $(seq ${L} -${_step} ${hL}); do
                ky=$(($L-$kx))
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                # done
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
        else
            echo "Data not removed"
        fi

        # Calculate the energy and spectral weight
        if [[ "$OnlyOnePoint" == "True" ]]; then
            kx=8
            ky=0
            echo "kx="${kx} "ky="${ky}
            python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
            --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
            --kx ${kx} --ky ${ky} \
            --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
            --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
        else
            # Correct Path:
            # M->X (pi,0)->(pi,pi)
            for ky in $(seq 0 ${_step} ${Lm1}); do
                kx=${L}
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
            # X->S (pi,pi)->(pi/2,pi/2)
            for kx in $(seq ${L} -${_step} ${hLp1}); do
                ky=${kx}
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
            # S->Gamma (pi/2,pi/2)->(0,0)
            for kx in $(seq ${hL} -${_step} 1); do
                ky=${kx}
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
            # Gamma->M (0,0)->(pi,0)
            for kx in $(seq 0 ${_step} ${Lm1}); do
                ky=0
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
            # M->S (pi,0)->(pi/2,pi/2)
            for kx in $(seq ${L} -${_step} ${hL}); do
                ky=$(($L-$kx))
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --j1 ${j1} --j2 ${j2} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            done
        fi
    fi

    if [[ "$runDraw" == "True" ]]; then
        #Draw the figure
        python -u graph.py ${datadir} True "h=${h} Kz=${Kz}" "[0,2,3,4,6,7], [r'\$M(\pi,0)\$', r'\$X(\pi,\pi)\$',r'\$S(\frac{\pi}{2},\frac{\pi}{2})\$', r'\$\Gamma(0,0)\$', r'\$M(\pi,0)\$', r'\$S(\frac{\pi}{2},\frac{\pi}{2})\$']"
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