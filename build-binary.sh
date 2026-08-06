#!/bin/bash
set -e

# Build the daemon-server binary for a given Linux architecture.
# This creates a standalone executable that bundles Python + all dependencies.
#
# Usage: build-binary.sh [amd64|arm64]   (default: amd64)
#
# The binary is a PyInstaller freeze, so it is architecture-specific.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH="${1:-amd64}"
case "$ARCH" in
    amd64) DOCKER_PLATFORM="linux/amd64"; MANYLINUX="quay.io/pypa/manylinux2014_x86_64" ;;
    arm64) DOCKER_PLATFORM="linux/arm64"; MANYLINUX="quay.io/pypa/manylinux2014_aarch64" ;;
    *) echo "ERROR: unsupported arch '$ARCH' (want amd64 or arm64)"; exit 1 ;;
esac

echo "=========================================="
echo "Building daemon-server binary (${DOCKER_PLATFORM})"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "Cleaning previous build for ${ARCH}..."
rm -rf build/ dist/daemon-server "dist/daemon-server-linux-${ARCH}" "dist/daemon-server-linux-${ARCH}.sha256"
mkdir -p dist

# Build binary using PyInstaller in Docker
echo "Building binary (this may take 2-3 minutes)..."
docker run --rm -i \
    --platform "$DOCKER_PLATFORM" \
    -v "$PWD:/src" \
    -w /src \
    "$MANYLINUX" bash -s <<'DOCKER_BUILD'
set -e
PYTHON_VERSION=3.12.8

# We build OpenSSL from source into our own prefix and link Python against that
# single copy. The base OS openssl (1.0.2k) is too old for Python 3.12's ssl
# module, and the distro's newer-openssl RPMs are not consistent across arches
# (the x86_64-only openssl11), while the image's bundled /opt/_internal/openssl
# ships libs without headers. Building it ourselves gives one self-contained,
# arch-agnostic source of truth. The exact version is immaterial -- any current
# OpenSSL 3.x satisfies Python's >=1.1.1 requirement -- so it is pinned only for
# reproducibility, not because a specific version is needed.
OPENSSL_VERSION=3.0.15
SSL_PREFIX=/opt/local-ssl

echo "Installing build deps..."
yum -y -q install libffi-devel bzip2-devel perl-core

echo "Building OpenSSL ${OPENSSL_VERSION} from source into ${SSL_PREFIX}..."
cd /tmp
curl -sSL "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz" -o openssl-src.tgz
tar xzf openssl-src.tgz
cd "openssl-${OPENSSL_VERSION}"
# --libdir=lib pins the layout (no lib64 ambiguity); shared libs so Python's ssl
# module links dynamically; rpath so the built libs find each other at runtime.
./Configure --prefix="${SSL_PREFIX}" --libdir=lib shared \
    "-Wl,-rpath,${SSL_PREFIX}/lib" >/tmp/openssl-configure.log 2>&1
make -j"$(nproc)" >/tmp/openssl-make.log 2>&1
# install_sw installs libraries + headers but skips man pages (faster, and we
# need only the dev files).
make install_sw >/tmp/openssl-install.log 2>&1

echo "Compiling Python ${PYTHON_VERSION} (shared) against ${SSL_PREFIX}..."
cd /tmp
curl -sSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o python-src.tgz
tar xzf python-src.tgz
cd "Python-${PYTHON_VERSION}"

# --enable-shared produces libpython3.12.so, which PyInstaller requires.
# --with-openssl points at the from-source prefix; --with-openssl-rpath=auto
# bakes that lib dir into the interpreter so import ssl resolves it at runtime.
./configure --enable-shared \
    --with-openssl="${SSL_PREFIX}" --with-openssl-rpath=auto >/tmp/configure.log 2>&1
make -j"$(nproc)" >/tmp/make.log 2>&1
make altinstall >/tmp/install.log 2>&1
ldconfig
cd /src

PY=/usr/local/bin/python3.12
"$PY" --version
# Fail closed unless the ssl module linked the OpenSSL we built from source
# (3.x), not the stale system 1.0.2k -- a silent fallback would ship a binary
# with broken TLS.
"$PY" - <<'PYCHECK'
import ssl, sys
v = ssl.OPENSSL_VERSION
print("ssl:", v)
if not v.startswith("OpenSSL 3."):
    sys.exit(f"ERROR: ssl linked {v!r}, expected the from-source OpenSSL 3.x")
PYCHECK
"$PY" -m ensurepip --upgrade
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q pyinstaller flask requests psutil waitress python-json-logger 'urllib3<2'
echo 'Running PyInstaller...'
"$PY" -m PyInstaller daemon-server.spec
echo 'Build complete!'
DOCKER_BUILD

# Check if build succeeded and save under the arch-specific asset name the
# hive-operator downloads (daemon-server-linux-<arch>).
if [ -f dist/daemon-server ]; then
    ASSET="dist/daemon-server-linux-${ARCH}"
    mv dist/daemon-server "$ASSET"
    SIZE=$(du -h "$ASSET" | cut -f1)
    echo ""
    echo "=========================================="
    echo "SUCCESS! Binary built successfully"
    echo "=========================================="
    echo "Location: $ASSET"
    echo "Size: $SIZE"
    echo "Platform: ${DOCKER_PLATFORM}"
    echo ""
    echo "Next steps:"
    echo "1. Test locally: docker run --rm --platform ${DOCKER_PLATFORM} -v \$PWD/dist:/app alpine /app/daemon-server-linux-${ARCH}"
    echo "2. Release: ./build-and-release.sh <version>"
else
    echo ""
    echo "ERROR: Build failed - binary not found at dist/daemon-server"
    exit 1
fi
