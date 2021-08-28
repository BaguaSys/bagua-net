NCCL_HOME:=../../build/
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I. -I./v4 -I./v3
PLUGIN_SO:=libnccl-net.so
NCCL_NET_V4:=./v4/nccl_net_v4.cc
NCCL_NET_V3:=./v3/nccl_net_v3.cc

default: $(PLUGIN_SO)

$(PLUGIN_SO): bagua_net.cc $(NCCL_NET_V4) $(NCCL_NET_V3)
	$(CXX) $(INC) \
	-std=c++17 -fPIC -shared \
	-o $@ $^ \
	-L../target/release/ -L../target/debug/ -l:libbagua_net.so

clean:
	rm -f $(PLUGIN_SO)
