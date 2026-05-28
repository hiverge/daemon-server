#!/bin/bash
set -e

# Build hive-webserver binary for Linux amd64
# This creates a standalone executable that bundles Python + all dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building hive-webserver binary (linux/amd64)"
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
    python:3.11-slim bash -c "
        echo 'Installing build dependencies...' && \
        apt-get update -qq && \
        apt-get install -y -qq binutils && \
        pip install -q pyinstaller flask requests psutil waitress && \
        echo 'Running PyInstaller...' && \
        pyinstaller hive-webserver.spec && \
        echo 'Build complete!'
    "

# Check if build succeeded and rename with architecture
if [ -f dist/hive-webserver ]; then
    mv dist/hive-webserver dist/hive-webserver-linux-amd64
    SIZE=$(du -h dist/hive-webserver-linux-amd64 | cut -f1)
    echo ""
    echo "=========================================="
    echo "SUCCESS! Binary built successfully"
    echo "=========================================="
    echo "Location: dist/hive-webserver-linux-amd64"
    echo "Size: $SIZE"
    echo "Platform: Linux x86_64 (amd64)"
    echo ""
    echo "Next steps:"
    echo "1. Test locally: docker run --rm -v \$PWD/dist:/app alpine /app/hive-webserver"
    echo "2. Build container: docker build -f Dockerfile.binary -t your-registry/hive-webserver:latest ."
    echo "3. Deploy: kubectl apply -f ../../deploy/webserver-binary-configmap.yaml"
else
    echo ""
    echo "ERROR: Build failed - binary not found at dist/hive-webserver"
    exit 1
fi
