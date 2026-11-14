# The Linux image that compiles FluidiTy's GPU code. It has no GPU: GPU code is compiled here, not run.
#
# The gate (tools/check_commit.py, step cuda) builds this image and runs it with the repository mounted read-only
# at /src and this checkout's named volume fluidity-cuda-out-<hash> at /out. Nothing is copied from the repository
# into the image.
# At run time, uv installs the project's tools (cmake, ninja, conan) from uv.lock into /out, and Conan keeps its
# packages there, so they are fetched once and reused by every later run.

# CUDA 13.4.1 with nvcc 13.4.59 and GCC 13 on Ubuntu 24.04, pinned by digest (tag 13.4.1-devel-ubuntu24.04).
FROM nvidia/cuda@sha256:a3f014424ccc86cf66c84d700d2896491424205dd9457e3116ed4f0e96381311

# uv 0.13.7, pinned by digest. It installs Python and the tools of pyproject.toml's dev group at run time.
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 \
    /uv /usr/local/bin/uv

WORKDIR /src
