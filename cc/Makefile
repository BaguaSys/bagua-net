INC:= -I. -I./v4 -I./v3
PLUGIN_SO:=libnccl-net.so
NCCL_NET_V4:=./v4/nccl_net_v4.cc
NCCL_NET_V3:=./v3/nccl_net_v3.cc

default: $(PLUGIN_SO)

$(PLUGIN_SO): bagua_net.cc $(NCCL_NET_V4) $(NCCL_NET_V3) ../target/release/libbagua_net.so
	$(CXX) -v $(INC) \
	-std=c++17 -fPIC -shared \
	-o $@ $^ \
	-L../target/release/ -l:libbagua_net.so

../target/release/libbagua_net.so:
	cargo build --release

test:
	cargo test --verbose

clean:
	rm -f $(PLUGIN_SO)
