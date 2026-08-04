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

# Clean previous build
echo "Cleaning previous build..."
rm -rf build/ dist/

# Build binary using PyInstaller in Docker
echo "Building binary (this may take 2-3 minutes)..."
docker run --rm -i \
    --platform "$DOCKER_PLATFORM" \
    -v "$PWD:/src" \
    -w /src \
    "$MANYLINUX" bash -s <<'DOCKER_BUILD'
set -e
PYTHON_VERSION=3.12.8

echo "Installing build deps and compiling Python ${PYTHON_VERSION} (shared) from source..."
yum -y -q install openssl11-devel libffi-devel bzip2-devel

# openssl11 uses a non-standard layout; assemble a standard prefix for --with-openssl
mkdir -p /tmp/ssl/include /tmp/ssl/lib
ln -sf /usr/include/openssl11/openssl /tmp/ssl/include/openssl
ln -sf /usr/lib64/openssl11/libssl.so /tmp/ssl/lib/libssl.so
ln -sf /usr/lib64/openssl11/libcrypto.so /tmp/ssl/lib/libcrypto.so

cd /tmp
curl -sSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o python-src.tgz
tar xzf python-src.tgz
cd "Python-${PYTHON_VERSION}"

# --enable-shared produces libpython3.12.so, which PyInstaller requires
./configure --enable-shared --with-openssl=/tmp/ssl --with-openssl-rpath=auto >/tmp/configure.log 2>&1
make -j"$(nproc)" >/tmp/make.log 2>&1
make altinstall >/tmp/install.log 2>&1
ldconfig
cd /src

PY=/usr/local/bin/python3.12
"$PY" --version
"$PY" -c 'import ssl; print("ssl:", ssl.OPENSSL_VERSION)'
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
    cp dist/daemon-server "$ASSET"
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
