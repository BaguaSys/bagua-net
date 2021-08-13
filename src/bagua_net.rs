use crate::utils;
use crate::utils::NCCLSocketDev;
use bytes::{Bytes, BytesMut};
use nix::sys::socket::{InetAddr, SockAddr};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{thread, time};
use thiserror::Error;

const NCCL_PTR_HOST: i32 = 1;
const NCCL_PTR_CUDA: i32 = 2;

type SocketListenCommID = usize;
type SocketSendCommID = usize;
type SocketRecvCommID = usize;
type SocketRequestID = usize;

pub struct BaguaNet {
    pub socket_devs: Vec<NCCLSocketDev>,
    pub listen_comm_next_id: usize,
    pub listen_comm_map: HashMap<SocketListenCommID, SocketListenComm>,
    pub send_comm_next_id: usize,
    pub send_comm_map: HashMap<SocketSendCommID, SocketSendComm>,
    pub recv_comm_next_id: usize,
    pub recv_comm_map: HashMap<SocketRecvCommID, SocketRecvComm>,
    pub socket_request_next_id: usize,
    pub socket_request_map: HashMap<SocketRequestID, SocketRequest>,

    pub send_sender: flume::Sender<(Arc<Mutex<net::TcpStream>>, Arc<Mutex<(bool, usize)>>, Bytes)>,
    pub send_receiver:
        flume::Receiver<(Arc<Mutex<net::TcpStream>>, Arc<Mutex<(bool, usize)>>, Bytes)>,
    pub recv_sender: flume::Sender<(
        Arc<Mutex<net::TcpStream>>,
        Arc<Mutex<(bool, usize)>>,
        &'static mut [u8],
    )>,
    pub recv_receiver: flume::Receiver<(
        Arc<Mutex<net::TcpStream>>,
        Arc<Mutex<(bool, usize)>>,
        &'static mut [u8],
    )>,

    send_worker: std::thread::JoinHandle<()>,
    recv_worker: std::thread::JoinHandle<()>,
}

#[derive(Debug)]
pub struct NCCLNetProperties {
    pub name: String,
    pub pci_path: String,
    pub guid: u64,
    pub ptr_support: i32, // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
    pub speed: i32,       // Port speed in Mbps.
    pub port: i32,
    pub max_comms: i32,
}

#[derive(Debug)]
pub struct SocketHandle {
    pub addr: nix::sys::socket::SockAddr,
}

pub struct SocketListenComm {
    pub tcp_listener: Arc<Mutex<net::TcpListener>>,
}

#[derive(Clone)]
pub struct SocketSendComm {
    pub tcp_stream: Arc<Mutex<net::TcpStream>>,
}

#[derive(Clone)]
pub struct SocketRecvComm {
    pub tcp_stream: Arc<Mutex<net::TcpStream>>,
    pub addr: net::SocketAddr,
}

#[derive(Error, Debug)]
pub enum BaguaNetError {
    #[error("io error")]
    IOError(String),
    #[error("tcp error")]
    TCPError(String),
}

pub struct SocketSendRequest {
    pub send_comm: SocketSendComm,
    pub data: Bytes,
    pub status: Arc<Mutex<(bool, usize)>>,
}

pub struct SocketRecvRequest {
    pub recv_comm: SocketRecvComm,
    pub status: Arc<Mutex<(bool, usize)>>,
}

pub enum SocketRequest {
    SendRequest(SocketSendRequest),
    RecvRequest(SocketRecvRequest),
}

impl BaguaNet {
    const DEFAULT_SOCKET_MAX_COMMS: i32 = 65536;

    pub fn new() -> Result<BaguaNet, BaguaNetError> {
        let (send_sender, send_receiver) = flume::unbounded();
        let (recv_sender, recv_receiver) = flume::unbounded();

        Ok(Self {
            socket_devs: utils::find_interfaces(),
            listen_comm_next_id: 0,
            listen_comm_map: Default::default(),
            send_comm_next_id: 0,
            send_comm_map: Default::default(),
            recv_comm_next_id: 0,
            recv_comm_map: Default::default(),
            socket_request_next_id: 0,
            socket_request_map: Default::default(),

            send_sender: send_sender.clone(),
            send_receiver: send_receiver.clone(),
            recv_sender: recv_sender.clone(),
            recv_receiver: recv_receiver.clone(),

            send_worker: std::thread::spawn(move || {
                for (tcp_stream, send_nbytes, data) in send_receiver.iter() {
                    let mut tcp_stream = tcp_stream.lock().unwrap();
                    // println!(
                    //     "start send task, data.len={}, local={:?}, peer={:?}",
                    //     data.len(),
                    //     (*tcp_stream).local_addr().unwrap(),
                    //     (*tcp_stream).peer_addr().unwrap()
                    // );

                    let send_size = data.len().to_be_bytes();
                    (*tcp_stream).write_all(&send_size[..]).unwrap();

                    if data.len() != 0 {
                        (*tcp_stream).write_all(&data[..]).unwrap();
                    }

                    // println!("write done, data={:02X?}", &data[..]);
                    (*send_nbytes.lock().unwrap()) = (true, data.len());
                }
            }),
            recv_worker: std::thread::spawn(move || {
                for (tcp_stream, recv_nbytes, mut data) in recv_receiver.iter() {
                    // NOTE: tcp_stream must be nonblock, otherwise it may deadlock
                    let mut target_nbytes = data.len().to_be_bytes();
                    (*tcp_stream.lock().unwrap()).set_nonblocking(false);
                    // (*tcp_stream.lock().unwrap())
                    //     .set_read_timeout(Some(Duration::from_secs_f64(3.)));
                    (*tcp_stream.lock().unwrap()).read_exact(&mut target_nbytes[..]).unwrap();
                    // match (*tcp_stream.lock().unwrap()).read_exact(&mut target_nbytes[..]) {
                    //     Ok(_) => {}
                    //     Err(err) => {
                    //         println!("read target_nbytes failed, err={:?}", err);
                    //         (*recv_nbytes.lock().unwrap()) = (true, 0);
                    //         continue;
                    //     }
                    // };
                    let target_nbytes = usize::from_be_bytes(target_nbytes);
                    // println!("recv task target_nbytes={}", target_nbytes);

                    let mut offset = 0;
                    while offset < target_nbytes {
                        {
                            let mut tcp_stream = tcp_stream.lock().unwrap();
                            tcp_stream.set_nonblocking(true);
                            // println!(
                            //     "start recv task, target_nbytes={}, offset={}, local={:?}, peer={:?}",
                            //     target_nbytes,
                            //     offset,
                            //     (*tcp_stream).local_addr().unwrap(),
                            //     (*tcp_stream).peer_addr().unwrap()
                            // );
                            let nbytes = match (*tcp_stream).read(&mut data[offset..target_nbytes]) {
                                Ok(nbytes) => {
                                    // if nbytes == 0 {
                                    //     println!("get zero bytes!!!!");
                                    // }
                                    nbytes
                                }
                                Err(err) => {
                                    // println!("recv task, err={:?}", err);
                                    if err.kind() != std::io::ErrorKind::WouldBlock {}

                                    0
                                }
                            };
                            // println!("recv data nbytes={}", nbytes);
                            offset += nbytes;
                            (*recv_nbytes.lock().unwrap()) = (false, offset);
                        }

                        std::thread::yield_now();
                        // thread::sleep(time::Duration::from_secs(3));
                    }

                    // println!("recv task done, bytes={:02X?}", &data[..offset]);
                    (*recv_nbytes.lock().unwrap()) = (true, offset);
                }
            }),
        })
    }

    pub fn devices(&self) -> Result<usize, BaguaNetError> {
        Ok(self.socket_devs.len())
    }

    pub fn get_properties(&self, dev_id: usize) -> Result<NCCLNetProperties, BaguaNetError> {
        let socket_dev = &self.socket_devs[dev_id];

        // println!(
        //     "get_properties dev_id={}, name={}, sockaddr={:?}, pci_path={}, speed={}",
        //     dev_id,
        //     socket_dev.interface_name.clone(),
        //     socket_dev.addr.clone().to_str(),
        //     socket_dev.pci_path.clone(),
        //     utils::get_net_if_speed(&socket_dev.interface_name),
        // );

        Ok(NCCLNetProperties {
            name: socket_dev.interface_name.clone(),
            pci_path: socket_dev.pci_path.clone(),
            guid: dev_id as u64,
            ptr_support: NCCL_PTR_HOST,
            speed: utils::get_net_if_speed(&socket_dev.interface_name),
            port: 0,
            max_comms: BaguaNet::DEFAULT_SOCKET_MAX_COMMS,
        })
    }

    pub fn listen(
        &mut self,
        dev_id: usize,
    ) -> Result<(SocketHandle, SocketListenCommID), BaguaNetError> {
        let socket_dev = &self.socket_devs[dev_id];
        // println!(
        //     "socket_dev={:?}, str={}",
        //     socket_dev,
        //     socket_dev.addr.clone().to_str()
        // );
        // println!(
        //     "go TcpListener::bind {:?}",
        //     socket_dev.addr.clone().to_str()
        // );
        let listener = match net::TcpListener::bind(socket_dev.addr.clone().to_str()) {
            Ok(listener) => {
                // println!(
                //     "TcpListener::bind {:?} ok",
                //     socket_dev.addr.clone().to_str()
                // );
                listener
            }
            Err(err) => return Err(BaguaNetError::TCPError(format!("{:?}", err))),
        };

        let socket_addr = listener.local_addr().unwrap();
        let socket_handle = SocketHandle {
            addr: SockAddr::new_inet(InetAddr::from_std(&socket_addr)),
        };

        let id = self.listen_comm_next_id;
        self.listen_comm_next_id += 1;
        self.listen_comm_map.insert(
            id,
            SocketListenComm {
                tcp_listener: Arc::new(Mutex::new(listener)),
            },
        );

        // println!(
        //     "listen id={}, dev_id={}, socket_addr={:?}",
        //     id, dev_id, socket_addr
        // );

        Ok((socket_handle, id))
    }

    pub fn connect(
        &mut self,
        _dev_id: usize,
        socket_handle: SocketHandle,
    ) -> Result<SocketSendCommID, BaguaNetError> {
        let stream = match net::TcpStream::connect(socket_handle.addr.clone().to_str()) {
            Ok(stream) => stream,
            Err(err) => {
                tracing::warn!(
                    "net::TcpStream::connect failed, err={:?}, socket_handle={:?}",
                    err,
                    socket_handle
                );
                // println!("connect addr={:?}", socket_handle.addr.clone().to_str());
                return Err(BaguaNetError::TCPError(format!(
                    "socket_handle={:?}, err={:?}",
                    socket_handle, err
                )));
            }
        };

        // stream
        //     .set_read_timeout(Some(Duration::from_secs_f64(30.)))
        //     .unwrap();
        // stream
        //     .set_write_timeout(Some(Duration::from_secs_f64(30.)))
        //     .unwrap();

        let id = self.send_comm_next_id;
        self.send_comm_next_id += 1;
        self.send_comm_map.insert(
            id,
            SocketSendComm {
                tcp_stream: Arc::new(Mutex::new(stream)),
            },
        );

        Ok(id)
    }

    pub fn accept(
        &mut self,
        listen_comm_id: SocketListenCommID,
    ) -> Result<SocketRecvCommID, BaguaNetError> {
        let listen_comm = self.listen_comm_map.get(&listen_comm_id).unwrap();
        let (stream, addr) = match listen_comm.tcp_listener.lock().unwrap().accept() {
            Ok(listen) => listen,
            Err(err) => {
                // println!(
                //     "accept failed, listen_comm_id={}, err={:?}",
                //     listen_comm_id, err
                // );
                return Err(BaguaNetError::TCPError(format!("{:?}", err)));
            }
        };

        // stream
        //     .set_read_timeout(Some(Duration::from_secs_f64(30.)))
        //     .unwrap();
        // stream
        //     .set_write_timeout(Some(Duration::from_secs_f64(30.)))
        //     .unwrap();

        let id = self.recv_comm_next_id;
        self.recv_comm_next_id += 1;
        self.recv_comm_map.insert(
            id,
            SocketRecvComm {
                tcp_stream: Arc::new(Mutex::new(stream)),
                addr: addr,
            },
        );

        Ok(id)
    }

    pub fn isend(
        &mut self,
        send_comm_id: SocketSendCommID,
        data: &'static [u8],
    ) -> Result<SocketRequestID, BaguaNetError> {
        let send_comm = self.send_comm_map.get(&send_comm_id).unwrap();
        // let mut send_stream = send_comm.tcp_stream.lock().unwrap();

        // println!(
        //     "isend data.len={}, send_comm_id={}, local={:?}, peer={:?}",
        //     data.len(),
        //     send_comm_id,
        //     (*send_stream).local_addr().unwrap(),
        //     (*send_stream).peer_addr().unwrap(),
        // );

        let id = self.socket_request_next_id;
        // println!(
        //     "isend id={}, data.len={}, send_comm_id={}",
        //     id,
        //     data.len(),
        //     send_comm_id
        // );

        self.socket_request_next_id += 1;
        let task_status = Arc::new(Mutex::new((false, 0)));
        self.socket_request_map.insert(
            id,
            SocketRequest::SendRequest(SocketSendRequest {
                send_comm: send_comm.clone(),
                data: Bytes::from_static(data),
                status: task_status.clone(),
            }),
        );

        let mut tcp_stream = send_comm.tcp_stream.clone();
        self.send_sender
            .send((tcp_stream, task_status, Bytes::from_static(data)))
            .unwrap();
        // std::thread::spawn(move || {
        //     (*tcp_stream.lock().unwrap()).write_all(&data[..]).unwrap();
        //     println!("write done, data.len={}", data.len());
        //     (*send_nbytes.lock().unwrap()) = data.len();
        // });

        Ok(id)
    }

    pub fn irecv(
        &mut self,
        recv_comm_id: SocketRecvCommID,
        data: &'static mut [u8],
    ) -> Result<SocketRequestID, BaguaNetError> {
        let recv_comm = self.recv_comm_map.get(&recv_comm_id).unwrap();
        // let mut recv_stream = recv_comm.tcp_stream.lock().unwrap();

        // unsafe {
        //     let ptr: *const u8 = &data[0];
        //     let mut k = Bytes::from_static(data);
        //     let ptr1: *const u8 = &k[0];

        //     let im_data: &'static [u8] = data;

        //     let k2 = BytesMut::from(im_data);
        //     let ptr2: *const u8 = &k2[0];
        //     println!("irecv ptr={:?}, ptr1={:?}, ptr2={:?}", ptr, ptr1, ptr2);
        // }
        // println!(
        //     "irecv data.len={}, recv_comm_id={}, local={:?}, peer={:?}",
        //     data.len(),
        //     recv_comm_id,
        //     (*recv_stream).local_addr().unwrap(),
        //     (*recv_stream).peer_addr().unwrap(),
        // );

        let id = self.socket_request_next_id;
        // println!(
        //     "irecv id={}, data.len={}, recv_comm_id={}",
        //     id,
        //     data.len(),
        //     recv_comm_id
        // );

        self.socket_request_next_id += 1;
        // (*recv_stream).read_exact(&mut data[..]).unwrap();
        let task_status = Arc::new(Mutex::new((false, 0)));
        self.socket_request_map.insert(
            id,
            SocketRequest::RecvRequest(SocketRecvRequest {
                recv_comm: recv_comm.clone(),
                status: task_status.clone(),
            }),
        );

        let mut tcp_stream = recv_comm.tcp_stream.clone();
        self.recv_sender
            .send((tcp_stream, task_status, data))
            .unwrap();
        // std::thread::spawn(move || {
        //     (*tcp_stream.lock().unwrap())
        //         .read_exact(&mut data[..])
        //         .unwrap();
        //     println!("read done, data.len={}", data.len());
        //     (*recv_nbytes.lock().unwrap()) = data.len();
        // });

        Ok(id)
    }

    pub fn test(&mut self, request_id: SocketRequestID) -> Result<(bool, usize), BaguaNetError> {
        let request = self.socket_request_map.get_mut(&request_id).unwrap();
        let ret = match request {
            SocketRequest::SendRequest(send_req) => {
                let task_status = send_req.status.lock().unwrap();
                // println!(
                //     "request_id={}, is_done={}, send_nbytes={}, send_req.data.len={}",
                //     request_id,
                //     (*task_status).0,
                //     (*task_status).1,
                //     send_req.data.len(),
                // );

                Ok(*task_status)
            }
            SocketRequest::RecvRequest(recv_req) => {
                let task_status = recv_req.status.lock().unwrap();
                // println!(
                //     "request_id={}, is_done={}, recv_nbytes={}, recv_req.data.len={}",
                //     request_id,
                //     (*task_status).0,
                //     (*task_status).1,
                //     recv_req.data.len(),
                // );

                Ok(*task_status)
            }
        };

        if let Ok(ret) = ret {
            if ret.0 {
                self.socket_request_map.remove(&request_id).unwrap();
            } else {
                std::thread::yield_now();
            }
        }

        ret
    }

    pub fn close_send(&mut self, send_comm_id: SocketSendCommID) -> Result<(), BaguaNetError> {
        self.send_comm_map.remove(&send_comm_id);

        Ok(())
    }

    pub fn close_recv(&mut self, recv_comm_id: SocketRecvCommID) -> Result<(), BaguaNetError> {
        self.recv_comm_map.remove(&recv_comm_id);

        Ok(())
    }

    pub fn close_listen(
        &mut self,
        listen_comm_id: SocketListenCommID,
    ) -> Result<(), BaguaNetError> {
        // println!("close_listen listen_comm_id={}", listen_comm_id);
        self.listen_comm_map.remove(&listen_comm_id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nix::sys::socket::{InetAddr, IpAddr, SockAddr};

    #[test]
    fn it_works() {
        let bagua_net = BaguaNet::new().unwrap();
        println!("bagua_net.socket_devs={:?}", bagua_net.socket_devs);

        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_socket_handle() {
        let addr = InetAddr::new(IpAddr::new_v4(127, 0, 0, 1), 8123);
        let socket_handle = SocketHandle {
            addr: SockAddr::new_inet(addr),
        };

        let addr = unsafe {
            let (c_sockaddr, _) = socket_handle.addr.as_ffi_pair();
            utils::from_libc_sockaddr(c_sockaddr).unwrap()
        };

        println!("socket_handle={:?}", addr.to_str());
    }
}
