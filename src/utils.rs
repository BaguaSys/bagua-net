use nix::net::if_::InterfaceFlags;
use nix::sys::socket::{SockAddr, InetAddr, AddressFamily};
use std::fs;

pub fn get_net_if_speed(device: &str) -> i32 {
    const DEFAULT_SPEED: i32 = 10000;

    let speed_path = format!("/sys/class/net/{}/speed", device);
    match fs::read_to_string(speed_path.clone()) {
        Ok(speed_str) => {
            return speed_str.parse::<i32>().unwrap_or(DEFAULT_SPEED);
        }
        Err(_) => {
            tracing::debug!(
                "Could not get speed from {}. Defaulting to 10 Gbps.",
                speed_path
            );
            DEFAULT_SPEED
        }
    }
}

#[derive(Debug)]
pub struct NCCLSocketDev {
    pub interface_name: String,
    pub addr: SockAddr,
    pub pci_path: String,
}

pub fn find_interfaces() -> Vec<NCCLSocketDev> {
    let mut socket_devs = Vec::<NCCLSocketDev>::new();
    const MAX_IF_NAME_SIZE: usize = 16;
    // TODO: support user specified interfaces
    let addrs = nix::ifaddrs::getifaddrs().unwrap();
    for ifaddr in addrs {
        match ifaddr.address {
            Some(addr) => {
                println!("interface {} address {}", ifaddr.interface_name, addr);

                if addr.family() != AddressFamily::Inet && addr.family() != AddressFamily::Inet6 {
                    continue;
                }

                if ifaddr.flags.contains(InterfaceFlags::IFF_LOOPBACK) {
                    continue;
                }

                assert_eq!(ifaddr.interface_name.len() < MAX_IF_NAME_SIZE, true);

                let found_ifs: Vec<&NCCLSocketDev> = socket_devs
                    .iter()
                    .filter(|scoket_dev| scoket_dev.interface_name == ifaddr.interface_name)
                    .collect();
                if found_ifs.len() > 0 {
                    continue;
                }

                socket_devs.push(NCCLSocketDev {
                    addr: addr,
                    interface_name: ifaddr.interface_name.clone(),
                    pci_path: format!("/sys/class/net/{}/device", ifaddr.interface_name),
                })
            }
            None => {
                println!(
                    "interface {} with unsupported address family",
                    ifaddr.interface_name
                );
            }
        }
    }

    socket_devs
}

/// Creates a `SockAddr` struct from libc's sockaddr.
///
/// Supports only the following address families: Unix, Inet (v4 & v6), Netlink and System.
/// Returns None for unsupported families.
///
/// # Safety
///
/// unsafe because it takes a raw pointer as argument.  The caller must
/// ensure that the pointer is valid.
#[cfg(not(target_os = "fuchsia"))]
pub(crate) unsafe fn from_libc_sockaddr(addr: *const libc::sockaddr) -> Option<SockAddr> {
    if addr.is_null() {
        None
    } else {
        match AddressFamily::from_i32(i32::from((*addr).sa_family)) {
            Some(AddressFamily::Unix) => None,
            Some(AddressFamily::Inet) => Some(SockAddr::Inet(
                InetAddr::V4(*(addr as *const libc::sockaddr_in)))),
            Some(AddressFamily::Inet6) => Some(SockAddr::Inet(
                InetAddr::V6(*(addr as *const libc::sockaddr_in6)))),
            // #[cfg(any(target_os = "android", target_os = "linux"))]
            // Some(AddressFamily::Netlink) => Some(SockAddr::Netlink(
            //     NetlinkAddr(*(addr as *const libc::sockaddr_nl)))),
            // #[cfg(any(target_os = "ios", target_os = "macos"))]
            // Some(AddressFamily::System) => Some(SockAddr::SysControl(
            //     SysControlAddr(*(addr as *const libc::sockaddr_ctl)))),
            // #[cfg(any(target_os = "android", target_os = "linux"))]
            // Some(AddressFamily::Packet) => Some(SockAddr::Link(
            //     LinkAddr(*(addr as *const libc::sockaddr_ll)))),
            // #[cfg(any(target_os = "dragonfly",
            //             target_os = "freebsd",
            //             target_os = "ios",
            //             target_os = "macos",
            //             target_os = "netbsd",
            //             target_os = "illumos",
            //             target_os = "openbsd"))]
            // Some(AddressFamily::Link) => {
            //     let ether_addr = LinkAddr(*(addr as *const libc::sockaddr_dl));
            //     if ether_addr.is_empty() {
            //         None
            //     } else {
            //         Some(SockAddr::Link(ether_addr))
            //     }
            // },
            // #[cfg(any(target_os = "android", target_os = "linux"))]
            // Some(AddressFamily::Vsock) => Some(SockAddr::Vsock(
            //     VsockAddr(*(addr as *const libc::sockaddr_vm)))),
            // Other address families are currently not supported and simply yield a None
            // entry instead of a proper conversion to a `SockAddr`.
            Some(_) | None => None,
        }
    }
}
