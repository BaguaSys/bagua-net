#include <netinet/in.h>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

#include "nccl_net_v4.h"
#include "bagua_net.h"

#define __hidden __attribute__((visibility("hidden")))

ncclDebugLogger_t NCCL_DEBUG_LOG;
#define NCCL_TRACE(FLAGS, ...) NCCL_DEBUG_LOG(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define NCCL_INFO(FLAGS, ...) NCCL_DEBUG_LOG(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define NCCL_WARN(...) NCCL_DEBUG_LOG(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)

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

__hidden ncclResult_t baguaNetInit(ncclDebugLogger_t logFunction)
{
    NCCL_DEBUG_LOG = logFunction;
    BaguaNet::instance();

    NCCL_TRACE(NCCL_ALL, "baguaNetInit!");

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetDevices(int *ndev)
{
    if (BaguaNet::instance().devices((int32_t *)ndev) != 0)
    {
        NCCL_WARN("baguaNetDevices failed, ndev=%d", *ndev);
        return ncclInternalError;
    }

    NCCL_TRACE(NCCL_ALL, "baguaNetDevices, ndev=%d", *ndev);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetGetProperties(int dev, ncclNetProperties_v4_t *props)
{
    NCCLNetPropertiesC inner_props;
    int ret = BaguaNet::instance().get_properties(dev, &inner_props);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetGetProperties failed, ret=%d, dev=%d", ret, dev);
        return ncclInternalError;
    }
    set_properties(*props, inner_props);
    NCCL_TRACE(NCCL_ALL, "baguaNetGetProperties, dev=%d", dev);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetListen(int dev, void *handle, void **listenComm)
{
    int ret = BaguaNet::instance().listen(dev, handle, listenComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetListen failed, ret=%d, dev=%d", ret, dev);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetListen, dev=%d", dev);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetConnect(int dev, void *handle, void **sendComm)
{
    int ret = BaguaNet::instance().connect(dev, handle, sendComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetConnect failed, ret=%d", ret);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetConnect ok, dev=%d", dev);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetAccept(void *listenComm, void **recvComm)
{
    int ret = BaguaNet::instance().accept(listenComm, recvComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetAccept failed, ret=%d", ret);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetAccept, listenComm=%p", listenComm);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetRegMr(void *comm, void *data, int size, int type, void **mhandle)
{
    NCCL_TRACE(NCCL_ALL, "baguaNetRegMr, comm=%p, data=%p, type=%d", comm, data, type);
    return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}

__hidden ncclResult_t baguaNetDeregMr(void *comm, void *mhandle)
{
    NCCL_TRACE(NCCL_ALL, "baguaNetDeregMr, comm=%p", comm);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetIsend(void *sendComm, void *data, int size, void *mhandle, void **request)
{
    int ret = BaguaNet::instance().isend(sendComm, data, size, mhandle, request);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetIsend failed, ret=%d, sendComm=%p, data=%p, size=%d", ret, sendComm, data, size);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetIsend, sendComm=%p, data=%p, size=%d, request_id=%d",
              sendComm, data, size, *(uintptr_t *)(*request));

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetIrecv(void *recvComm, void *data, int size, void *mhandle, void **request)
{
    int ret = BaguaNet::instance().irecv(recvComm, data, size, mhandle, request);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetIrecv failed, ret=%d, sendComm=%p, data=%p, size=%d", ret, recvComm, data, size);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetIrecv, recvComm=%p, data=%p, size=%d, request_id=%d",
              recvComm, data, size, *(uintptr_t *)(*request));

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetFlush(void *recvComm, void *data, int size, void *mhandle, void **request)
{
    // We don't support CUDA pointers, so we don't need a flush operation
    return ncclInternalError;
}

__hidden ncclResult_t baguaNetTest(void *request, int *done, int *size)
{
    bool b_done = false;
    uintptr_t nbytes = 0;
    int ret = BaguaNet::instance().test(request, &b_done, &nbytes);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetTest failed, ret=%d, request_id=%d",
                  ret, *static_cast<uintptr_t *>(request));
        return ncclInternalError;
    }
    *done = b_done ? 1 : 0;
    if (b_done && size != NULL)
    {
        *size = nbytes;
        NCCL_TRACE(NCCL_ALL, "size=%d", *size);
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetTest, request_id=%d, done=%d, nbytes=%d",
              *static_cast<uintptr_t *>(request), *done, nbytes);

    return ncclSuccess;
}

__hidden ncclResult_t baguaNetCloseSend(void *sendComm)
{
    int ret = BaguaNet::instance().close_send(sendComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetCloseSend failed, ret=%d, sendComm=%p", ret, sendComm);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetCloseSend, sendComm=%p", sendComm);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetCloseRecv(void *recvComm)
{
    int ret = BaguaNet::instance().close_recv(recvComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetCloseRecv failed, ret=%d, recvComm=%p", ret, recvComm);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetCloseRecv, recvComm=%p", recvComm);
    return ncclSuccess;
}

__hidden ncclResult_t baguaNetCloseListen(void *listenComm)
{
    int ret = BaguaNet::instance().close_listen(listenComm);
    if (ret != 0)
    {
        NCCL_WARN("baguaNetCloseListen failed, ret=%d, listenComm=%p", ret, listenComm);
        return ncclInternalError;
    }
    NCCL_TRACE(NCCL_ALL, "baguaNetCloseListen, listenComm=%p", listenComm);
    return ncclSuccess;
}

ncclNet_v4_t ncclNetPlugin_v4 = {
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
