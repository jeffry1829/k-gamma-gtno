#!/bin/bash
for i in `seq 1.0 0.1 2.0`; do
    for j in `seq 0 24 119`; do
        if [ ${i} = "1.0" ] && [ ${j} = "0" ]; then
            continue
        fi
        echo ${i} ${j}
        julia aniso-k-1d.jl ${i} ${j} --threads `nproc`
    done
done