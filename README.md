---
modified: 2021-08-04T13:41:18.756Z
---

## Install

```bash
cargo build
cd cc && make

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(readlink -f .):$(readlink -f ../target/debug)
```
