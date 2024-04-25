#!/bin/bash

while true
do
    python optim_k_gamma_LG.py --CTMARGS_ctm_conv_tol=1e-3 --params_out=test --GLOBALARGS_dtype complex128 --tiling 2SITE --GLOBALARGS_device cuda:0
done