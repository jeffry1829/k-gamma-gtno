#!/bin/bash
hx=2.5
chi=16
bond_dim=2
_size=11
statefile="D=2TFIM_output_state.json"
_device="cpu"
_dtype="complex128"

rm guptri_excitedE.txt

# Calculate the energy and spectral weight
# M->X
# for kx in $(seq 24 24); do
for ky in $(seq 0 2 23); do
    kx=24
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0
done
# done
# X->S
for kx in $(seq 24 -2 13); do
    # for ky in $(seq 24 -2 13); do
    ky=$kx
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0
    # done
done
# S->gamma
for kx in $(seq 12 -2 1); do
    ky=$kx
    kx=${kx}.0
    ky=${ky}.0
    # for ky in $(seq 12 -2 1); do
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0
    # done
done
# gamma->M
# for kx in $(seq 12 -2 1); do
for kx in $(seq 0 2 23); do
    ky=0
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0
    # done
done
# M->S
# for kx in $(seq 12 -2 1); do
for kx in $(seq 24 -2 12); do
    ky=$((24-$kx))
    kx=${kx}.0
    ky=${ky}.0
    echo "kx="${kx} "ky="${ky}
    python test_guptri.py ${kx} ${ky} ${hx} 0
    # done
done

#Draw the figure
python guptri_graph.py
