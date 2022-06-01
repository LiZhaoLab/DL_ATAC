# The evolution and mutational robustness of chromatin accessibility in _Drosophila_

## Neural Network

For implementation of the neural network,
please see the [dedicated folder](CNN_attn/README.md).

## Orthology mapping

For implementation of the orthology mapping,
please see the [dedicated folder](comp_genomics/README.md).

## Environment

The conda environment used for this project
is described in [`conda-env.txt`](conda-env.txt).

This project was run on the Rockefeller University
High Performance Computing cluster,
using a variety of Nvidia K80, P100, V100, and V100S GPUs.

The Nvidia drivers were as follows:
```
NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2
```

The OS details were as follows:
```
$ lsb_release -a
LSB Version:    :core-4.1-amd64:core-4.1-ia32:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-ia32:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-ia32:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch
Distributor ID: RedHatEnterpriseServer
Description:    Red Hat Enterprise Linux Server release 7.8 (Maipo)
Release:        7.8
Codename:       Maipo
$ uname -r
3.10.0-1127.el7.x86_64
```
