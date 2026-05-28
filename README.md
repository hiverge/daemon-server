# Daemon Server Binary

Standalone binary distribution of the daemon-server for fast deployment via init containers.

## Overview

The daemon-server is a Flask-based web service that runs inside sandbox containers to execute user code and health checks. This binary distribution eliminates the need for Kaniko image building (3-5 minutes) by providing a pre-built executable that can be injected into any base image.

**Startup time reduction: 3-5 minutes → 5-10 seconds**

## Architecture Support

Currently supports:
- **Linux x86_64 (amd64)** - `daemon-server-linux-amd64`

The operator only deploys workloads to `linux/amd64` nodes, so only one binary is needed.

## Building the Binary

### Prerequisites
- Docker (with BuildKit support)
- 2-3 minutes build time

### Build Command

```bash
./build-binary.sh
```

This script:
1. Cleans previous builds
2. Uses Docker with `--platform linux/amd64` to ensure correct architecture
3. Installs PyInstaller and dependencies (flask, requests, psutil)
4. Creates a standalone 64-bit ELF executable (~12MB)
5. Outputs to `dist/daemon-server-linux-amd64`

### What's Inside the Binary

The binary bundles:
- Python 3.11 interpreter
- Flask web framework
- Waitress WSGI server
- requests library
- psutil library
- All Python standard library modules
- Daemon server application code (main.py, common_tools.py)

## File Naming Convention

Binary filenames include the architecture to support potential multi-platform builds:

```
daemon-server-{os}-{arch}
```

Current: `daemon-server-linux-amd64`

Future (if needed):
- `daemon-server-linux-arm64` (for ARM-based nodes)
- `daemon-server-darwin-amd64` (for local testing on Intel Macs)
- `daemon-server-darwin-arm64` (for local testing on Apple Silicon)

## Verification

Check binary properties:

```bash
# File type
file dist/daemon-server-linux-amd64
# Output: ELF 64-bit LSB executable, x86-64

# Size
du -h dist/daemon-server-linux-amd64
# Output: ~12M

# Verify checksum
sha256sum -c dist/daemon-server-linux-amd64.sha256
```

Test locally:

```bash
# Run binary in container
docker run --rm \
  -e PORT=8080 \
  -v $PWD/dist:/bin \
  -p 8080:8080 \
  python:3.11-slim /bin/daemon-server-linux-amd64
# Should start webserver on port 8080

# Test health endpoint (in another terminal)
curl http://localhost:8080/health
# Should return: {"status":"healthy"}
```

## Why This Approach?

**Before (Kaniko):**
1. User specifies base image (e.g., `python:3.11-slim`)
2. Operator creates Kaniko job to build new image with webserver
3. Build takes 3-5 minutes
4. New image pushed to registry
5. Sandbox pod uses built image

**After (Binary Injection):**
1. User specifies base image
2. Init container downloads/copies pre-built binary (5-10 seconds)
3. Binary injected into user's base image via shared volume
4. Sandbox pod starts immediately

**Benefits:**
- **95% faster startup:** 5-10 seconds vs 3-5 minutes
- **No image registry pollution:** No intermediate images created
- **Works with any base image:** As long as it's Linux x86_64
- **Simpler pipeline:** No Kaniko job orchestration
- **Better resource utilization:** No build resources needed per experiment

## Dependencies

The binary is self-contained and has no external dependencies. It works with any Linux x86_64 base image, including:
- Minimal images (alpine, distroless)
- Python images (python:3.x)
- Ubuntu/Debian images
- Custom user images

No Python installation or pip packages required in the base image.

## Troubleshooting

### Binary won't execute

Check file permissions:
```bash
chmod +x /path/to/daemon-server-linux-amd64
```

### "No such file or directory" on execution

The base image might be missing required libraries. Check with:
```bash
ldd daemon-server-linux-amd64
```

Most Linux images include required libraries (glibc), but very minimal images like `FROM scratch` won't work.

### Wrong architecture

Verify the binary architecture matches node architecture:
```bash
kubectl get nodes -o wide
# Look at ARCH column, should be amd64
```

## Development

To modify the webserver:

1. Edit `main.py`
2. Rebuild: `./build-binary.sh`
3. Test locally (see Verification section)
4. Release: `./build-and-release.sh v1.x.x`
5. Update operator to use new version
