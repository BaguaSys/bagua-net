use crate::utils;
use crate::utils::NCCLSocketDev;
use bytes::{BufMut, BytesMut};
use nix::sys::socket::{InetAddr, SockAddr};
use opentelemetry::{
    metrics::{BoundValueRecorder, ObserverResult},
    trace::{Span, TraceContextExt, Tracer},
    KeyValue,
};
use socket2::{Domain, Socket, Type};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

const NCCL_PTR_HOST: i32 = 1;
const NCCL_PTR_CUDA: i32 = 2;

lazy_static! {
    static ref HANDLER_ALL: [KeyValue; 1] = [KeyValue::new("handler", "all")];
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

// TODO: make Rotating communicator
#[derive(Clone)]
pub struct SocketSendComm {
    pub msg_sender: mpsc::UnboundedSender<(&'static [u8], Arc<Mutex<RequestState>>)>,
}

#[derive(Clone)]
pub struct SocketRecvComm {
    pub msg_sender: mpsc::UnboundedSender<(&'static mut [u8], Arc<Mutex<RequestState>>)>,
}

#[derive(Error, Debug, Clone)]
pub enum BaguaNetError {
    #[error("io error")]
    IOError(String),
    #[error("tcp error")]
    TCPError(String),
    #[error("inner error")]
    InnerError(String),
}

pub struct SocketSendRequest {
    pub state: Arc<Mutex<RequestState>>,
    pub trace_span: opentelemetry::global::BoxedSpan,
}

pub struct SocketRecvRequest {
    pub state: Arc<Mutex<RequestState>>,
    pub trace_span: opentelemetry::global::BoxedSpan,
}

#[derive(Debug)]
pub struct RequestState {
    pub nsubtasks: usize,
    pub completed_subtasks: usize,
    pub nbytes_transferred: usize,
    pub err: Option<BaguaNetError>,
}

pub enum SocketRequest {
    SendRequest(SocketSendRequest),
    RecvRequest(SocketRecvRequest),
}

static TELEMETRY_INIT_ONCE: std::sync::Once = std::sync::Once::new();
// static TELEMETRY_GUARD: Option<TelemetryGuard> = None;

struct AppState {
    exporter: opentelemetry_prometheus::PrometheusExporter,
    isend_nbytes_gauge: BoundValueRecorder<'static, u64>,
    irecv_nbytes_gauge: BoundValueRecorder<'static, u64>,
    isend_per_second: Arc<Mutex<f64>>,
    isend_nbytes_per_second: Arc<Mutex<f64>>,
    isend_percentage_of_effective_time: Arc<Mutex<f64>>,
    // isend_nbytes_gauge: BoundValueRecorder<'static, u64>,
    // irecv_nbytes_gauge: BoundValueRecorder<'static, u64>,
    uploader: std::thread::JoinHandle<()>,
}

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
    pub trace_span_context: opentelemetry::Context,
    pub trace_on_flag: bool,
    pub rank: i32,
    state: Arc<AppState>,
    nstreams: usize,
    task_split_threshold: usize,
    tokio_rt: tokio::runtime::Runtime,
}

impl BaguaNet {
    const DEFAULT_SOCKET_MAX_COMMS: i32 = 65536;
    const DEFAULT_LISTEN_BACKLOG: i32 = 16384;
    const DEFAULT_N_CONTROL_STREAM: i32 = 2;

    pub fn new() -> Result<BaguaNet, BaguaNetError> {
        let rank: i32 = std::env::var("RANK")
            .unwrap_or("-1".to_string())
            .parse()
            .unwrap();
        TELEMETRY_INIT_ONCE.call_once(|| {
            if rank == -1 || rank > 7 {
                return;
            }

            let jaeger_addr = match std::env::var("BAGUA_NET_JAEGER_ADDRESS") {
                Ok(jaeger_addr) => {
                    tracing::info!("detected auto tuning server, connecting");
                    jaeger_addr
                }
                Err(_) => {
                    tracing::warn!("Jaeger server not detected.");
                    return;
                }
            };

            opentelemetry::global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
            opentelemetry_jaeger::new_pipeline()
                .with_collector_endpoint(format!("http://{}/api/traces", jaeger_addr))
                .with_service_name("bagua-net")
                .install_batch(opentelemetry::runtime::AsyncStd)
                .unwrap();
        });

        let tracer = opentelemetry::global::tracer("bagua-net");
        let mut span = tracer.start(format!("BaguaNet-{}", rank));
        span.set_attribute(KeyValue::new(
            "socket_devs",
            format!("{:?}", utils::find_interfaces()),
        ));

        let prom_exporter = opentelemetry_prometheus::exporter()
            .with_default_histogram_boundaries(vec![16., 1024., 4096., 1048576.])
            .init();

        let isend_nbytes_per_second = Arc::new(Mutex::new(0.));
        let isend_percentage_of_effective_time = Arc::new(Mutex::new(0.));
        let isend_per_second = Arc::new(Mutex::new(0.));

        let meter = opentelemetry::global::meter("bagua-net");
        let isend_nbytes_per_second_clone = isend_nbytes_per_second.clone();
        meter
            .f64_value_observer(
                "isend_nbytes_per_second",
                move |res: ObserverResult<f64>| {
                    res.observe(
                        *isend_nbytes_per_second_clone.lock().unwrap(),
                        HANDLER_ALL.as_ref(),
                    );
                },
            )
            .init();
        let isend_per_second_clone = isend_per_second.clone();
        meter
            .f64_value_observer("isend_per_second", move |res: ObserverResult<f64>| {
                res.observe(
                    *isend_per_second_clone.lock().unwrap(),
                    HANDLER_ALL.as_ref(),
                );
            })
            .init();
        let isend_percentage_of_effective_time_clone = isend_percentage_of_effective_time.clone();
        meter
            .f64_value_observer(
                "isend_percentage_of_effective_time",
                move |res: ObserverResult<f64>| {
                    res.observe(
                        *isend_percentage_of_effective_time_clone.lock().unwrap(),
                        HANDLER_ALL.as_ref(),
                    );
                },
            )
            .init();
        let state = Arc::new(AppState {
            exporter: prom_exporter.clone(),
            isend_nbytes_gauge: meter
                .u64_value_recorder("isend_nbytes")
                .init()
                .bind(HANDLER_ALL.as_ref()),
            irecv_nbytes_gauge: meter
                .u64_value_recorder("irecv_nbytes")
                .init()
                .bind(HANDLER_ALL.as_ref()),
            isend_per_second: isend_per_second,
            isend_nbytes_per_second: isend_nbytes_per_second,
            isend_percentage_of_effective_time: isend_percentage_of_effective_time,
            uploader: std::thread::spawn(move || {
                let prometheus_addr =
                    std::env::var("BAGUA_NET_PROMETHEUS_ADDRESS").unwrap_or_default();
                let (user, pass, address) = match utils::parse_user_pass_and_addr(&prometheus_addr)
                {
                    Some(ret) => ret,
                    None => return,
                };

                loop {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                    let metric_families = prom_exporter.registry().gather();
                    match prometheus::push_metrics(
                        "BaguaNet",
                        prometheus::labels! { "rank".to_owned() => rank.to_string(), },
                        &address,
                        metric_families,
                        Some(prometheus::BasicAuthentication {
                            username: user.clone(),
                            password: pass.clone(),
                        }),
                    ) {
                        Ok(_) => {}
                        Err(err) => {
                            tracing::warn!("{:?}", err);
                        }
                    }
                }
            }),
        });

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
            trace_span_context: opentelemetry::Context::current_with_span(span),
            rank: rank,
            trace_on_flag: rank < 8,
            state: state,
            nstreams: std::env::var("BAGUA_NET_NSTREAMS")
                .unwrap_or("2".to_owned())
                .parse()
                .unwrap(),
            task_split_threshold: std::env::var("BAGUA_NET_TASK_SPLIT_THRESHOLD")
                .unwrap_or("1048576".to_owned())
                .parse()
                .unwrap(),
            tokio_rt: tokio::runtime::Runtime::new().unwrap(),
        })
    }

    pub fn devices(&self) -> Result<usize, BaguaNetError> {
        Ok(self.socket_devs.len())
    }

    pub fn get_properties(&self, dev_id: usize) -> Result<NCCLNetProperties, BaguaNetError> {
        let socket_dev = &self.socket_devs[dev_id];

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
        let addr = match socket_dev.addr.clone() {
            SockAddr::Inet(inet_addr) => inet_addr,
            others => {
                return Err(BaguaNetError::InnerError(format!(
                    "Got invalid socket address, which is {:?}",
                    others
                )))
            }
        };

        let socket = match Socket::new(
            match addr {
                InetAddr::V4(_) => Domain::IPV4,
                InetAddr::V6(_) => Domain::IPV6,
            },
            Type::STREAM,
            None,
        ) {
            Ok(sock) => sock,
            Err(err) => return Err(BaguaNetError::IOError(format!("{:?}", err))),
        };
        socket.bind(&addr.to_std().into()).unwrap();
        socket.listen(BaguaNet::DEFAULT_LISTEN_BACKLOG).unwrap();

        let listener: net::TcpListener = socket.into();
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

        Ok((socket_handle, id))
    }

    pub fn connect(
        &mut self,
        _dev_id: usize,
        socket_handle: SocketHandle,
    ) -> Result<SocketSendCommID, BaguaNetError> {
        let mut streams_input = Vec::new();
        for _ in 0..self.nstreams {
            let stream = match net::TcpStream::connect(socket_handle.addr.clone().to_str()) {
                Ok(stream) => stream,
                Err(err) => {
                    tracing::warn!(
                        "net::TcpStream::connect failed, err={:?}, socket_handle={:?}",
                        err,
                        socket_handle
                    );
                    return Err(BaguaNetError::TCPError(format!(
                        "socket_handle={:?}, err={:?}",
                        socket_handle, err
                    )));
                }
            };

            let (msg_sender, mut msg_receiver) =
                mpsc::unbounded_channel::<(&'static [u8], Arc<Mutex<RequestState>>)>();
            // let metrics = self.state.clone();
            // TODO: Consider dynamically assigning tasks to make the least stream full
            self.tokio_rt.spawn(async move {
                let mut stream = tokio::net::TcpStream::from_std(stream).unwrap();
                stream.set_nodelay(true).unwrap();
                loop {
                    let (data, state) = match msg_receiver.recv().await {
                        Some(it) => it,
                        None => break,
                    };

                    if let Err(err) = stream.write_all(&data[..]).await {
                        state.lock().unwrap().err =
                            Some(BaguaNetError::InnerError(format!("{:?}", err)));
                        break;
                    }

                    match state.lock() {
                        Ok(mut state) => {
                            state.completed_subtasks += 1;
                            state.nbytes_transferred += data.len();
                        }
                        Err(poisoned) => {
                            tracing::warn!("{:?}", poisoned);
                        }
                    };
                }
            });
            streams_input.push(msg_sender);
        }

        let ctrl_stream = match net::TcpStream::connect(socket_handle.addr.clone().to_str()) {
            Ok(ctrl_stream) => ctrl_stream,
            Err(err) => {
                tracing::warn!(
                    "net::TcpStream::connect failed, err={:?}, socket_handle={:?}",
                    err,
                    socket_handle
                );
                return Err(BaguaNetError::TCPError(format!(
                    "socket_handle={:?}, err={:?}",
                    socket_handle, err
                )));
            }
        };
        println!("{:?} connect to {:?}", ctrl_stream.local_addr(), ctrl_stream.peer_addr());

        let (msg_sender, mut msg_receiver) = tokio::sync::mpsc::unbounded_channel();
        let task_split_threshold = self.task_split_threshold;
        let id = self.send_comm_next_id;
        self.send_comm_next_id += 1;
        let metrics = self.state.clone();
        let send_comm = SocketSendComm {
            msg_sender: msg_sender,
        };
        self.tokio_rt.spawn(async move {
            let mut ctrl_stream = tokio::net::TcpStream::from_std(ctrl_stream).unwrap();
            ctrl_stream.set_nodelay(true).unwrap();
            let out_timer = std::time::Instant::now();
            let mut sum_in_time = 0.;
            let mut downstream_id = 0;
            loop {
                let (data, state) = match msg_receiver.recv().await {
                    Some(it) => it,
                    None => break,
                };
                let in_timer = std::time::Instant::now();
                // let send_nbytes = data.len().to_be_bytes();
                // ctrl_stream.write_all(&send_nbytes[..]).await.unwrap();
                ctrl_stream.write_u32(data.len() as u32).await.unwrap();
                println!("send to {:?} target_nbytes={}", ctrl_stream.peer_addr(), data.len());

                if data.len() != 0 {
                    let bucket_size =
                        if data.len() >= task_split_threshold && data.len() > streams_input.len() {
                            data.len() + (streams_input.len() - 1) / streams_input.len()
                        } else {
                            data.len()
                        };
                    for bucket in data.chunks(bucket_size) {
                        state.lock().unwrap().nsubtasks += 1;
                        streams_input[downstream_id]
                            .send((bucket, state.clone()))
                            .unwrap();
                        downstream_id = (downstream_id + 1) % streams_input.len();
                    }
                }
                state.lock().unwrap().completed_subtasks += 1;

                // let dur = in_timer.elapsed().as_secs_f64();
                // sum_in_time += dur;

                // *metrics.isend_per_second.lock().unwrap() = 1. / dur;
                // *metrics.isend_percentage_of_effective_time.lock().unwrap() =
                //     sum_in_time / out_timer.elapsed().as_secs_f64();
            }
        });
        self.send_comm_map.insert(id, send_comm);

        Ok(id)
    }

    pub fn accept(
        &mut self,
        listen_comm_id: SocketListenCommID,
    ) -> Result<SocketRecvCommID, BaguaNetError> {
        let listen_comm = self.listen_comm_map.get(&listen_comm_id).unwrap();
        let mut streams_input = Vec::new();
        for _ in 0..self.nstreams {
            let (stream, _addr) = match listen_comm.tcp_listener.lock().unwrap().accept() {
                Ok(listen) => listen,
                Err(err) => {
                    return Err(BaguaNetError::TCPError(format!("{:?}", err)));
                }
            };

            let (msg_sender, mut msg_receiver) =
                mpsc::unbounded_channel::<(&'static mut [u8], Arc<Mutex<RequestState>>)>();
            let metrics = self.state.clone();
            self.tokio_rt.spawn(async move {
                let mut stream = tokio::net::TcpStream::from_std(stream).unwrap();
                stream.set_nodelay(true).unwrap();
                loop {
                    let (data, state) = match msg_receiver.recv().await {
                        Some(it) => it,
                        None => break,
                    };

                    stream.read_exact(&mut data[..]).await.unwrap();
                    // metrics.irecv_nbytes_gauge.record(data.len() as u64);

                    match state.lock() {
                        Ok(mut state) => {
                            state.completed_subtasks += 1;
                            state.nbytes_transferred += data.len();
                        }
                        Err(poisoned) => {
                            tracing::warn!("{:?}", poisoned);
                        }
                    };
                }
            });
            streams_input.push(msg_sender);
        }

        let (ctrl_stream, _addr) = match listen_comm.tcp_listener.lock().unwrap().accept() {
            Ok(listen) => listen,
            Err(err) => {
                return Err(BaguaNetError::TCPError(format!("{:?}", err)));
            }
        };

        println!("listen on local_addr={:?}", ctrl_stream.local_addr());

        let (msg_sender, mut msg_receiver) = mpsc::unbounded_channel();
        let task_split_threshold = self.task_split_threshold;
        let id = self.recv_comm_next_id;
        self.recv_comm_next_id += 1;
        let recv_comm = SocketRecvComm {
            msg_sender: msg_sender,
        };
        self.tokio_rt.spawn(async move {
            let mut ctrl_stream = tokio::net::TcpStream::from_std(ctrl_stream).unwrap();
            ctrl_stream.set_nodelay(true).unwrap();
            let mut downstream_id = 0;
            loop {
                let (data, state) = match msg_receiver.recv().await {
                    Some(it) => it,
                    None => break,
                };
                let target_nbytes = match ctrl_stream.read_u32().await {
                    Ok(nbytes) => nbytes as usize,
                    Err(err) => {
                        state.lock().unwrap().err =
                        Some(BaguaNetError::InnerError(format!("{:?}", err)));
                        return;
                    }
                };
                // let mut target_nbytes = data.len().to_be_bytes();
                // if let Err(err) = ctrl_stream.read_exact(&mut target_nbytes[..]).await {
                //     state.lock().unwrap().err =
                //         Some(BaguaNetError::InnerError(format!("{:?}", err)));
                //     break;
                // }
                // let target_nbytes = usize::from_be_bytes(target_nbytes);

                println!("{:?} recv target_nbytes={}", ctrl_stream.local_addr(), target_nbytes);

                if target_nbytes == 0 {
                    state.lock().unwrap().completed_subtasks += 1;
                } else if target_nbytes > data.len() {
                    match state.lock() {
                        Ok(mut state) => {
                            state.err = Some(BaguaNetError::InnerError(format!(
                                "Invalid target_nbytes!! target_nbytes={}, buff.len={}",
                                target_nbytes,
                                data.len(),
                            )));
                            break;
                        }
                        Err(poisoned) => {
                            tracing::warn!("{:?}", poisoned);
                        }
                    };
                } else {
                    let bucket_size = if target_nbytes >= task_split_threshold
                        && target_nbytes > streams_input.len()
                    {
                        target_nbytes + (streams_input.len() - 1) / streams_input.len()
                    } else {
                        target_nbytes
                    };

                    for bucket in data[..target_nbytes].chunks_mut(bucket_size) {
                        state.lock().unwrap().nsubtasks += 1;
                        streams_input[downstream_id]
                            .send((&mut bucket[..], state.clone()))
                            .unwrap();
                        downstream_id = (downstream_id + 1) % streams_input.len();
                    }
                    state.lock().unwrap().completed_subtasks += 1;
                }
            }
        });
        self.recv_comm_map.insert(id, recv_comm);

        Ok(id)
    }

    pub fn isend(
        &mut self,
        send_comm_id: SocketSendCommID,
        data: &'static [u8],
    ) -> Result<SocketRequestID, BaguaNetError> {
        let tracer = opentelemetry::global::tracer("bagua-net");
        let mut span = tracer
            .span_builder(format!("isend-{}", send_comm_id))
            .with_parent_context(self.trace_span_context.clone())
            .start(&tracer);
        let send_comm = self.send_comm_map.get(&send_comm_id).unwrap();
        let id = self.socket_request_next_id;

        span.set_attribute(KeyValue::new("id", id as i64));
        span.set_attribute(KeyValue::new("nbytes", data.len() as i64));

        self.socket_request_next_id += 1;
        let task_state = Arc::new(Mutex::new(RequestState {
            nsubtasks: 1,
            completed_subtasks: 0,
            nbytes_transferred: 0,
            err: None,
        }));
        self.socket_request_map.insert(
            id,
            SocketRequest::SendRequest(SocketSendRequest {
                state: task_state.clone(),
                trace_span: span,
            }),
        );

        send_comm.msg_sender.send((data, task_state)).unwrap();

        Ok(id)
    }

    pub fn irecv(
        &mut self,
        recv_comm_id: SocketRecvCommID,
        data: &'static mut [u8],
    ) -> Result<SocketRequestID, BaguaNetError> {
        let tracer = opentelemetry::global::tracer("bagua-net");
        let mut span = tracer
            .span_builder(format!("irecv-{}", recv_comm_id))
            .with_parent_context(self.trace_span_context.clone())
            .start(&tracer);
        let recv_comm = self.recv_comm_map.get(&recv_comm_id).unwrap();
        let id = self.socket_request_next_id;

        span.set_attribute(KeyValue::new("id", id as i64));

        self.socket_request_next_id += 1;
        let task_state = Arc::new(Mutex::new(RequestState {
            nsubtasks: 1,
            completed_subtasks: 0,
            nbytes_transferred: 0,
            err: None,
        }));
        self.socket_request_map.insert(
            id,
            SocketRequest::RecvRequest(SocketRecvRequest {
                state: task_state.clone(),
                trace_span: span,
            }),
        );

        recv_comm.msg_sender.send((data, task_state)).unwrap();

        Ok(id)
    }

    pub fn test(&mut self, request_id: SocketRequestID) -> Result<(bool, usize), BaguaNetError> {
        let request = self.socket_request_map.get_mut(&request_id).unwrap();
        let ret = match request {
            SocketRequest::SendRequest(send_req) => {
                let state = send_req.state.lock().unwrap();
                if let Some(err) = state.err.clone() {
                    return Err(err);
                }

                let task_completed = state.nsubtasks == state.completed_subtasks;
                if task_completed {
                    send_req.trace_span.end();
                }
                // println!("send_req.state={:?}", state);
                Ok((task_completed, state.nbytes_transferred))
            }
            SocketRequest::RecvRequest(recv_req) => {
                let state = recv_req.state.lock().unwrap();
                if let Some(err) = state.err.clone() {
                    return Err(err);
                }

                let task_completed = state.nsubtasks == state.completed_subtasks;
                if task_completed {
                    recv_req.trace_span.end();
                }
                // println!("recv_req.state={:?}", state);
                Ok((task_completed, state.nbytes_transferred))
            }
        };

        if let Ok(ret) = ret {
            if ret.0 {
                self.socket_request_map.remove(&request_id).unwrap();
            }
        }

        ret
    }

    pub fn close_send(&mut self, send_comm_id: SocketSendCommID) -> Result<(), BaguaNetError> {
        self.send_comm_map.remove(&send_comm_id);
        println!("close_send send_comm_id={}", send_comm_id);

        Ok(())
    }

    pub fn close_recv(&mut self, recv_comm_id: SocketRecvCommID) -> Result<(), BaguaNetError> {
        self.recv_comm_map.remove(&recv_comm_id);
        println!("close_recv recv_comm_id={}", recv_comm_id);

        Ok(())
    }

    pub fn close_listen(
        &mut self,
        listen_comm_id: SocketListenCommID,
    ) -> Result<(), BaguaNetError> {
        self.listen_comm_map.remove(&listen_comm_id);

        Ok(())
    }
}

impl Drop for BaguaNet {
    fn drop(&mut self) {
        // TODO: make shutdown global
        self.trace_span_context.span().end();
        opentelemetry::global::shutdown_tracer_provider();
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
