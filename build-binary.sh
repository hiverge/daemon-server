#!/bin/bash
set -e

# Build hive-webserver binary for Linux amd64
# This creates a standalone executable that bundles Python + all dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building daemon-server binary (linux/amd64)"
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
docker run --rm \
    --platform linux/amd64 \
    -v "$PWD:/src" \
    -w /src \
    quay.io/pypa/manylinux2014_x86_64 bash -c "
        echo 'Installing Python 3.8 (shared) and build deps...' && \
        yum -y -q --disablerepo='epel*' install rh-python38-python-devel && \
        source /opt/rh/rh-python38/enable && \
        python3 -m ensurepip --upgrade && \
        python3 -m pip install -q --upgrade pip && \
        python3 -m pip install -q pyinstaller flask requests psutil waitress python-json-logger 'urllib3<2' && \
        echo 'Running PyInstaller...' && \
        pyinstaller daemon-server.spec && \
        echo 'Build complete!'
    "

# Check if build succeeded and rename with architecture
if [ -f dist/daemon-server ]; then
    cp dist/daemon-server dist/daemon-server-linux-amd64
    SIZE=$(du -h dist/daemon-server | cut -f1)
    echo ""
    echo "=========================================="
    echo "SUCCESS! Binary built successfully"
    echo "=========================================="
    echo "Location: dist/daemon-server"
    echo "Also saved as: dist/daemon-server-linux-amd64"
    echo "Size: $SIZE"
    echo "Platform: Linux x86_64 (amd64)"
    echo ""
    echo "Next steps:"
    echo "1. Test locally: docker run --rm -v \$PWD/dist:/app alpine /app/daemon-server"
    echo "2. Build container: docker build -f Dockerfile.binary -t your-registry/daemon-server:latest ."
else
    echo ""
    echo "ERROR: Build failed - binary not found at dist/daemon-server"
    exit 1
fi
