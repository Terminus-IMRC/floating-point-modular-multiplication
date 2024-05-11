# syntax=docker/dockerfile:1

ARG UBUNTU_RELEASE=noble


FROM ubuntu:${UBUNTU_RELEASE} AS benchmark-builder

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      ca-certificates \
      cmake \
      curl \
      g++ \
      make

ARG BENCHMARK_VERSION=1.8.5

RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp \
 && curl -L "https://github.com/google/benchmark/archive/refs/tags/v${BENCHMARK_VERSION}.tar.gz" -o benchmark.tar.gz \
 && tar xzf benchmark.tar.gz \
 && cd "benchmark-${BENCHMARK_VERSION}/" \
 && cmake . \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/benchmark \
      -D BENCHMARK_ENABLE_TESTING=OFF \
      -D BENCHMARK_INSTALL_DOCS=OFF \
 && cmake --build . --target install


FROM ubuntu:${UBUNTU_RELEASE} AS sde-builder

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      ca-certificates \
      curl \
      xz-utils

ARG SDE_URL=https://downloadmirror.intel.com/823664/sde-external-9.38.0-2024-04-18-lin.tar.xz
ENV SDE_VERSION=${SDE_URL##*/sde-external-}
ENV SDE_VERSION=${SDE_VERSION%.tar.xz}

RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp \
 && curl -L "${SDE_URL}" -o sde.tar.xz \
 && tar xJf sde.tar.xz \
 && cp -r "sde-external-${SDE_VERSION}/" /opt/sde


FROM ubuntu:${UBUNTU_RELEASE} AS intel-repository-key-downloader

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      ca-certificates \
      curl \
      gpg

RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp \
 && curl -L 'https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB' -o intel-sw-products.pub \
 && gpg --dearmor --output /usr/share/keyrings/intel-sw-products.gpg intel-sw-products.pub


FROM ubuntu:${UBUNTU_RELEASE}

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      ca-certificates

COPY --from=intel-repository-key-downloader /usr/share/keyrings/intel-sw-products.gpg /usr/share/keyrings/intel-sw-products.gpg

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    echo 'deb [signed-by=/usr/share/keyrings/intel-sw-products.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/intel-oneapi.list \
 && apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      clang \
      clang-format \
      cmake \
      cmake-format \
      g++ \
      git \
      intel-oneapi-compiler-dpcpp-cpp \
      less \
      make

COPY --from=benchmark-builder /opt/benchmark /opt/benchmark
COPY --from=sde-builder /opt/sde /opt/sde
