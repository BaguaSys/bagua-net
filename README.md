## Install

Install libnccl-net.so.

```bash
cargo build
cd cc && make

export BAGUA_NET_PATH=$(readlink -f .):$(readlink -f ../target/debug)
```

## Test

Run nccl-test to test performance.

```bash
# install nccl and nccl-test
git clone https://github.com/NVIDIA/nccl.git && cd nccl && git checkout v2.10.3-1
make -j src.build && make install
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# run nccl baseline
mpirun \
  --allow-run-as-root \
  -H ${HOST1}:1,${HOST2}:1 --np 2 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth01 \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# run nccl with bagua-net
mpirun \
  --allow-run-as-root \
  -H ${HOST1}:1,${HOST2}:1 --np 2 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth01 \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BAGUA_NET_PATH \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

If the installation is successful, there will be a log like this `NCCL INFO Using network BaguaNet`
