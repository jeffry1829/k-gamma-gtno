
export MKL_NUM_THREADS=1

# python main.py --chi 16 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 1 #--GLOBALARGS_device -1 --CTMARGS_projector_svd_method "GESVD" 

# python testbenchmark.py --chi 16 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 'cpu' --CTMARGS_projector_svd_method GESDD
# python -m cProfile -o output.pstats testbenchmark.py --chi 16 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 'cuda:2' --CTMARGS_projector_svd_method GESDD
#python -m gprof2dot -f pstats output.pstats | dot -T png -o profile.png
for chi in {16,32,64,128,256} 
do
    echo "chi: ${chi}"
    python testbenchmark.py --chi ${chi} --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 'cuda:0' --CTMARGS_projector_svd_method GESDD
done

# python main.py --chi 32 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0 #--CTMARGS_projector_svd_method GESVD
# python main.py --chi 16 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0 #--CTMARGS_projector_svd_method GESVD
# python main.py --chi 64 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --CTMARGS_projector_svd_method "GESVD" --GLOBALARGS_device 0

# python main.py --chi 128 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --CTMARGS_projector_svd_method "GESVD" --GLOBALARGS_device 0

# python main.py --chi 16 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0

# python main.py --chi 32 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0

# python main.py --chi 64 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device 0

# python main.py --chi 128 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10 --GLOBALARGS_device -1

# python main.py --chi 256 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 50 --CTMARGS_projector_svd_method "GESVD" --GLOBALARGS_device 0


# python main.py --chi 16 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 32 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 64 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 128 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 16 --bondim 5 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 32 --bondim 5 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 64 --bondim 5 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 128 --bondim 4 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 128 --bondim 5 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10

# python main.py --chi 128 --bondim 2 --CTMARGS_ctm_conv_tol 0 --CTMARGS_ctm_max_iter 10


# export MKL_NUM_THREADS=1
# python test.py

# export CYTNX_INC=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
# export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
# export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"
# export CYTNX_CXXFLAGS="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")"

# nvcc -I${CYTNX_INC}  -DUNI_HPTT -DUNI_CUTENSOR -I/home/j9263178//libcutensor-linux-x86_64-1.7.0.1-archive/include -DUNI_CUQUANTUM -I/home/j9263178//cuquantum-linux-x86_64-23.06.1.8_cuda11-archive/include -DUNI_GPU -I/home/j9263178/miniconda3/envs/cytnx/include  test.cpp ${CYTNX_LIB} /home/j9263178/libcutensor-linux-x86_64-1.7.0.1-archive/lib/12/libcutensor.so /home/j9263178/libcutensor-linux-x86_64-1.7.0.1-archive/lib/12/libcutensorMg.so -ldl /home/j9263178/cuquantum-linux-x86_64-23.06.1.8_cuda11-archive/lib/libcutensornet.so /home/j9263178/cuquantum-linux-x86_64-23.06.1.8_cuda11-archive/lib/libcustatevec.so -ldl -L/home/j9263178/miniconda3/envs/cytnx/lib /home/j9263178/miniconda3/envs/cytnx/lib/libcusolver.so /home/j9263178/miniconda3/envs/cytnx/lib/libcurand.so /home/j9263178/miniconda3/envs/cytnx/lib/libcublas.so /home/j9263178/miniconda3/envs/cytnx/lib/libcudart_static.a -ldl -lrt -lcudadevrt /home/j9263178/miniconda3/envs/cytnx/lib/libcusparse.so /home/j9263178/miniconda3/envs/cytnx/lib/libmkl_rt.so  -lpthread -lm -ldl -lpthread -lm -ldl  /home/j9263178/Cytnx_ctmrg/hptt/lib/libhptt.a -o test
# ./test