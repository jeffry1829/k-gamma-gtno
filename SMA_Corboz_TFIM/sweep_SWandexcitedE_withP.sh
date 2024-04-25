#!/bin/bash
#!/bin/bash
for hx in 2.5
# for h in 0.00 0.05 0.10 0.15
do
    bond_dim=2
    chi=8
    _size=1
    _step=4
    _device="cuda:1"
    # _dtype="complex128"
    _dtype="float64"
    statefile="floatD=2TFIM_output_state.json"
    datadir="data/withP_floatTFIM_hx${hx}chi${chi}/"
    extra_flags="--CTMARGS_ctm_force_dl True --MultiGPU True --CTMARGS_projector_eps_multiplet 1e-4 --CTMARGS_ctm_conv_tol 1e-8"
    extra_flags=${extra_flags}" --CTMARGS_projector_svd_reltol 1e-8"
    
    mkdir -p ${datadir}
    cp ${datadir}../${statefile} ${datadir}
    reuseCTMRGenv="True"
    removeCTMRGenv="False"
    SMAMethod="SMA_withP.py"
    StoredMatMethod="SMA_stored_mat_withP.py"
    runSMA="True"
    runStoredMat="True"
    runDraw="True"

    OnlyOnePoint="True"
    max_iter=100000000

    # mkdir -p ${datadir}

    if [[ "$runSMA" == "True" ]]; then
        if [[ "$OnlyOnePoint" == "True" ]]; then
            kx=0
            ky=0
            echo "kx="${kx} "ky="${ky}
            python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
            --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
            --kx ${kx} --ky ${ky} \
            --hx ${hx} --datadir ${datadir} ${extra_flags}
        else
            # M->X
            for ky in $(seq 0 ${_step} 23); do
                kx=24
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # X->S
            for kx in $(seq 24 -${_step} 13); do
                ky=$kx
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # S->gamma
            for kx in $(seq 12 -${_step} 1); do
                ky=$kx
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # gamma->M
            for kx in $(seq 0 ${_step} 23); do
                ky=0
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # M->S
            for kx in $(seq 24 -${_step} 12); do
                ky=$((24-$kx))
                echo "kx="${kx} "ky="${ky}
                python -u ${SMAMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} \
                --hx ${hx} --datadir ${datadir} ${extra_flags}
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
            echo "Data removed"
        else
            echo "Data not removed"
        fi

        # Calculate the energy and spectral weight
        if [[ "$OnlyOnePoint" == "True" ]]; then
            kx=0
            ky=0
            echo "kx="${kx} "ky="${ky}
            python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
        else
            # M->X
            # for kx in $(seq 24 24); do
            for ky in $(seq 0 ${_step} 23); do
                kx=24
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # done
            # X->S
            for kx in $(seq 24 -${_step} 13); do
                ky=$kx
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # S->gamma
            for kx in $(seq 12 -${_step} 1); do
                ky=$kx
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # gamma->M
            for kx in $(seq 0 ${_step} 23); do
                ky=0
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
            # M->S
            for kx in $(seq 24 -${_step} 12); do
                ky=$((24-$kx))
                echo "kx="${kx} "ky="${ky}
                python -u ${StoredMatMethod} --GLOBALARGS_dtype ${_dtype} --bond_dim ${bond_dim} --chi ${chi} \
                --statefile ${statefile} --size ${_size} --CTMARGS_ctm_max_iter ${max_iter} --GLOBALARGS_device ${_device} \
                --kx ${kx} --ky ${ky} --hx ${hx} --datadir ${datadir} ${extra_flags}
            done
        fi
    fi

    if [[ "$runDraw" == "True" ]]; then
        #Draw the figure
        python -u graph.py ${datadir} True "h=${h} Kz=${Kz}" "[0, 4, 8, 16], [r'\$M(\pi,0)\$', r'\$\Gamma(0,0)\$',r'\$K(\pi,\frac{\pi}{2})\$', r'\$M(\pi,0)\$']"
    fi
done