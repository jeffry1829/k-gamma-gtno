#!/bin/bash
#
# First parameter is the hx
# Second parameter is the q
# Third parameter is the datadir
# Fourth parameter is the device
# Fifth parameter is the runSMA
# Sixth parameter is the runStoredMat
# Seventh parameter is the runDraw
# Eighth parameter is the eig_size
# Nineth parameter is the chi
for q in $2
do
    hx=$1
    # q=${q}
    chi="$9"
    bond_dim=4
    _size=2
    L=$((2*${_size}+2))
    Lm1=$((L-1))
    hL=$((L/2))
    hLp1=$((hL+1))
    _step=3
    _device="$4"
    # _device="cpu"
    _dtype="complex128"
    statefile="TFIMD${bond_dim}chi${chi}${_dtype}hx${hx}q${q}.json"
    datadir="$3/TFIM_withP_hx${hx}q${q}chi${chi}size${_size}bonddim${bond_dim}dtype${_dtype}/"
    mkdir -p ${datadir}
    cp ${datadir}../${statefile} ${datadir}
    reuseCTMRGenv="True"
    removeCTMRGenv="False"
    extra_flags="--CTMARGS_ctm_force_dl True --MultiGPU False --CTMARGS_projector_eps_multiplet 1e-12 --CTMARGS_ctm_conv_tol 3e-6"
    extra_flags=${extra_flags}" --NormMat True --HamiMat True --CTMARGS_projector_svd_reltol 1e-12"
    extra_flags=${extra_flags}" --CTMARGS_projector_method 4X4 --CTMARGS_projector_svd_method GESDD_CPU"
    extra_flags=${extra_flags}" --CTMARGS_ctm_env_init_type CTMRG --UseVUMPSansazAC False"
    extra_flags=${extra_flags}" --CTMARGS_ctm_absorb_normalization inf"
    SMAMethod="SMA_TFIM_gpugraph_divide_withP.py"
    StoredMatMethod="SMA_stored_mat_TFIM.py"
    runSMA="$5"
    runStoredMat="$6"
    runDraw="$7"
    runGUPTRI="False"
    runDrawGUPTRI="False"
    skipCalculatedPoints="True"
    eig_size="$8"

    OnlyOnePoint="False"
    # max_iter=10
    max_iter=100000000

    # mkdir -p ${datadir}

    if [[ "$runSMA" == "True" ]]; then
        if [[ "$OnlyOnePoint" == "True" ]]; then
            kx=${L}
            ky=0
            echo "kx="${kx} "ky="${ky}
            if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                echo "Not skipped"
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
            fi
        else
            # Correct Path:
            # M->X (pi,0)->(pi,pi)
            for ky in $(seq 0 ${_step} ${Lm1}); do
                kx=${L}
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
            done
            # X->S (pi,pi)->(pi/2,pi/2)
            for kx in $(seq ${L} -${_step} ${hLp1}); do
                ky=${kx}
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
            done
            # S->Gamma (pi/2,pi/2)->(0,0)
            for kx in $(seq ${hL} -${_step} 1); do
                ky=${kx}
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
                # done
            done
            # Gamma->M (0,0)->(pi,0)
            for kx in $(seq 0 ${_step} ${Lm1}); do
                ky=0
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
                # done
            done
            # M->S (pi,0)->(pi/2,pi/2)
            for kx in $(seq ${L} -${_step} ${hL}); do
                ky=$(($L-$kx))
                # for ky in $(seq 12 -2 1); do
                echo "kx="${kx} "ky="${ky}
                if [ ! -f "${datadir}kx${kx}.0ky${ky}.0HamiMat.npy" -a $skipCalculatedPoints == "True" ]; then
                    echo "Not skipped"
                    python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv}
                fi
                # done
            done
        fi
    fi

    if [[ "$runStoredMat" == "True" ]]; then
        # Check if the user really wants to remove the data
        echo "Are you sure you want to remove the data? (y/n) (input within 3 secs)"
        read -t 3 answer
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
                --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv} --eig_size ${eig_size}
            else
                # Correct Path:
                # M->X (pi,0)->(pi,pi)
                for ky in $(seq 0 ${_step} ${Lm1}); do
                    kx=${L}
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv} --eig_size ${eig_size}
                done
                # X->S (pi,pi)->(pi/2,pi/2)
                for kx in $(seq ${L} -${_step} ${hLp1}); do
                    ky=${kx}
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv} --eig_size ${eig_size}
                done
                # S->Gamma (pi/2,pi/2)->(0,0)
                for kx in $(seq ${hL} -${_step} 1); do
                    ky=${kx}
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv} --eig_size ${eig_size}
                done
                # Gamma->M (0,0)->(pi,0)
                for kx in $(seq 0 ${_step} ${Lm1}); do
                    ky=0
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv} --eig_size ${eig_size}
                done
                # M->S (pi,0)->(pi/2,pi/2)
                for kx in $(seq ${L} -${_step} ${hL}); do
                    ky=$(($L-$kx))
                    echo "kx="${kx} "ky="${ky}
                    python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                    --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                    --kx ${kx} --ky ${ky} \
                    --hx ${hx} --q ${q} --datadir ${datadir} ${extra_flags} \
                    --reuseCTMRGenv ${reuseCTMRGenv} --removeCTMRGenv ${removeCTMRGenv} --eig_size ${eig_size}
                done
            fi
        else
            echo "Data not removed and not running"
        fi
    fi

    if [[ "$runDraw" == "True" ]]; then
        #Draw the figure
        python -u graph.py ${datadir} True "hx=${hx}, q=${q}" "[0,2,3,4,6,7], [r'\$M(\pi,0)\$', r'\$X(\pi,\pi)\$',r'\$S(\frac{\pi}{2},\frac{\pi}{2})\$', r'\$\Gamma(0,0)\$', r'\$M(\pi,0)\$', r'\$S(\frac{\pi}{2},\frac{\pi}{2})\$']"\
        "${eig_size}" "./data/figures/TFIM_fixedCTM_D${bond_dim}chi${chi}L${L}h${hx}"
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