/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <nccl.h>
#include <nccl_net.h>
#include <netinet/in.h>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include <memory>
#include <functional>
#include <iostream>
#include <vector>

#include "bagua_net.h"

#define __hidden __attribute__((visibility("hidden")))

void set_properties(ncclNetProperties_v4_t &lhs, NCCLNetPropertiesC &rhs)
{
    lhs.name = const_cast<char *>(rhs.name);
    lhs.pciPath = const_cast<char *>(rhs.pci_path);
    lhs.guid = rhs.guid;
    lhs.ptrSupport = rhs.ptr_support;
    lhs.speed = rhs.speed;
    lhs.port = rhs.port;
    lhs.maxComms = rhs.max_comms;
}

class BagueNet
{
public:
    static BagueNet &instance()
    {
        static BagueNet instance;
        return instance;
    }
    BagueNet(BagueNet const &) = delete;
    void operator=(BagueNet const &) = delete;

    int32_t devices(int32_t *ndev)
    {
        return bagua_net_c_devices(inner.get(), ndev);
    }

    int32_t get_properties(int32_t dev_id, ncclNetProperties_v4_t *props)
    {
        // HINT: The name and pci_path of ncclNetProperties_v4_t are both
        // references and cannot be passed in directly
        auto& inner_props = device_props.at(dev_id);
        if (!inner_props) {
            inner_props = std::shared_ptr<NCCLNetPropertiesC>(
                new NCCLNetPropertiesC{.name = NULL, .pci_path = NULL},
                [](NCCLNetPropertiesC* ptr) {
                    delete ptr->name;
                    delete ptr->pci_path;
                }
            );
            int ret = bagua_net_c_get_properties(inner.get(), dev_id, inner_props.get());
            if (ret != 0)
            {
                return ret;
            }
        }

        set_properties(*props, *inner_props);
        return 0;
    }

    int32_t listen(int32_t dev_id, void *handle, void **listen_comm)
    {
        auto socket_listen_comm_id = std::make_unique<uintptr_t>(0);
        int32_t ret = bagua_net_c_listen(inner.get(), dev_id, static_cast<SocketHandleC *>(handle), socket_listen_comm_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *listen_comm = socket_listen_comm_id.release();
        return 0;
    }

    int32_t connect(int32_t dev_id, void *handle, void **send_comm)
    {
        auto socket_send_comm_id = std::make_unique<uintptr_t>(0);
        int32_t ret = bagua_net_c_connect(inner.get(), dev_id, static_cast<SocketHandleC *>(handle), socket_send_comm_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *send_comm = socket_send_comm_id.release();
        return 0;
    }

    int32_t accept(void *listen_comm, void **recv_comm)
    {
        uintptr_t listen_comm_id = *static_cast<uintptr_t *>(listen_comm);
        auto recv_comm_id = std::make_unique<uintptr_t>(0);
        int32_t ret = bagua_net_c_accept(inner.get(), listen_comm_id, recv_comm_id.get());
        if (ret != 0)
        {
            return ret;
        }

        *recv_comm = recv_comm_id.release();
        return 0;
    }

    int32_t close_listen(void *listen_comm)
    {
        auto listen_comm_id = std::unique_ptr<uintptr_t>(static_cast<uintptr_t *>(listen_comm));
        return bagua_net_c_close_listen(inner.get(), *listen_comm_id);
    }

private:
    BagueNet()
    {
        inner = std::unique_ptr<BaguaNetC, std::function<void(BaguaNetC *)> >(
            bagua_net_c_create(),
            [](BaguaNetC *ptr)
            {
                bagua_net_c_destroy(&ptr);
            });
        int32_t ndev = -1;
        if (bagua_net_c_devices(inner.get(), &ndev) == 0 && ndev != -1) {
            device_props.resize(ndev);
        }
    }

private:
    std::unique_ptr<BaguaNetC, std::function<void(BaguaNetC *)> > inner;
    std::vector<std::shared_ptr<NCCLNetPropertiesC>> device_props;
};

__hidden ncclResult_t baguaNetInit(ncclDebugLogger_t logFunction)
{
    std::cerr << "baguaNetInit" << std::endl;
    BagueNet::instance();
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetDevices(int *ndev)
{
    if (BagueNet::instance().devices((int32_t *)ndev) != 0)
    {
        std::cerr << "baguaNetDevices failed, ndev=" << *ndev << std::endl;
        return ncclInternalError;
    }

    std::cerr << "baguaNetDevices, ndev=" << *ndev << std::endl;
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetGetProperties(int dev, ncclNetProperties_v4_t *props)
{
    std::cerr << "baguaNetGetProperties, dev=" << dev
              << std::endl;
    if (BagueNet::instance().get_properties(dev, props) != 0)
    {
        return ncclInternalError;
    }

    std::cerr << "props->name=" << props->name
              << std::endl;

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetListen(int dev, void *handle, void **listenComm)
{
    std::cerr << "baguaNetListen, dev=" << dev
              << std::endl;
    int ret = BagueNet::instance().listen(dev, handle, listenComm);
    if (ret != 0)
    {
        return ncclInternalError;
    }

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetConnect(int dev, void *handle, void **sendComm)
{
    std::cerr << "baguaNetConnect, dev=" << dev
              << std::endl;
    int ret = BagueNet::instance().connect(dev, handle, sendComm);
    if (ret != 0)
    {
        return ncclInternalError;
    }

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetAccept(void *listenComm, void **recvComm)
{
    std::cerr << "baguaNetAccept"
              << std::endl;
    int ret = BagueNet::instance().accept(listenComm, recvComm);
    if (ret != 0)
    {
        return ncclInternalError;
    }

    return ncclSuccess;
}
__hidden ncclResult_t baguaNetRegMr(void *comm, void *data, int size, int type, void **mhandle) { return ncclInternalError; }
__hidden ncclResult_t baguaNetDeregMr(void *comm, void *mhandle) { return ncclInternalError; }
__hidden ncclResult_t baguaNetIsend(void *sendComm, void *data, int size, void *mhandle, void **request) { return ncclInternalError; }
__hidden ncclResult_t baguaNetIrecv(void *recvComm, void *data, int size, void *mhandle, void **request) { return ncclInternalError; }
__hidden ncclResult_t baguaNetFlush(void *recvComm, void *data, int size, void *mhandle, void **request) { return ncclInternalError; }
__hidden ncclResult_t baguaNetTest(void *request, int *done, int *size) { return ncclInternalError; }
__hidden ncclResult_t baguaNetCloseSend(void *sendComm) { return ncclInternalError; }
__hidden ncclResult_t baguaNetCloseRecv(void *recvComm) { return ncclInternalError; }
__hidden ncclResult_t baguaNetCloseListen(void *listenComm)
{

    std::cerr << "baguaNetCloseListen"
              << std::endl;
    int ret = BagueNet::instance().close_listen(listenComm);
    if (ret != 0)
    {
        return ncclInternalError;
    }
    return ncclSuccess;
}

ncclNet_t ncclNetPlugin_v4 = {
    "BaguaNet",
    baguaNetInit,
    baguaNetDevices,
    baguaNetGetProperties,
    baguaNetListen,
    baguaNetConnect,
    baguaNetAccept,
    baguaNetRegMr,
    baguaNetDeregMr,
    baguaNetIsend,
    baguaNetIrecv,
    baguaNetFlush,
    baguaNetTest,
    baguaNetCloseSend,
    baguaNetCloseRecv,
    baguaNetCloseListen};
