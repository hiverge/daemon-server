# Hive Webserver Binary

Standalone binary distribution of the hive-webserver for fast deployment via init containers.

## Overview

The hive-webserver is a Flask-based web service that runs inside sandbox containers to execute user code and health checks. This binary distribution eliminates the need for Kaniko image building (3-5 minutes) by providing a pre-built executable that can be injected into any base image.

**Startup time reduction: 3-5 minutes → 5-10 seconds**

## Architecture Support

Currently supports:
- **Linux x86_64 (amd64)** - `hive-webserver-linux-amd64`

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
5. Outputs to `dist/hive-webserver-linux-amd64`

### What's Inside the Binary

The binary bundles:
- Python 3.11 interpreter
- Flask web framework
- requests library
- psutil library
- All Python standard library modules
- Webserver application code (main.py)

## Distribution Options

### Option 1: GitHub Releases (Recommended for Public Repos)

Release the binary to GitHub where it can be downloaded via HTTPS:

```bash
./build-and-release.sh v1.0.0
```

This creates a GitHub release with:
- Binary: `hive-webserver-linux-amd64`
- Checksum: `hive-webserver-linux-amd64.sha256`
- Download instructions
- Usage documentation

Download URL format:
```
https://github.com/YOUR_ORG/YOUR_REPO/releases/download/v1.0.0/hive-webserver-linux-amd64
```

**Note:** For private repositories, GitHub releases require authentication. See Option 2 or 3 for private repos.

### Option 2: Container Registry (DockerHub/GCR/ECR)

Package the binary in a minimal container for use as init container source:

```bash
# Build container
docker build -f Dockerfile.binary -t your-registry/hive-webserver-binary:v1.0.0 .

# Push to registry
docker push your-registry/hive-webserver-binary:v1.0.0
```

The `FROM scratch` image is <15MB and contains only the binary.

### Option 3: Cloud Storage (S3/GCS)

Upload the binary to cloud storage:

```bash
# AWS S3
aws s3 cp dist/hive-webserver-linux-amd64 s3://your-bucket/binaries/v1.0.0/

# Google Cloud Storage
gsutil cp dist/hive-webserver-linux-amd64 gs://your-bucket/binaries/v1.0.0/
```

## Kubernetes Integration

### Using Init Container with GitHub Releases

The experiment controller injects the binary using an init container that downloads it via curl:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sandbox-pod
spec:
  initContainers:
  - name: inject-webserver
    image: curlimages/curl:latest
    command:
    - sh
    - -c
    - |
      curl -L -o /shared/bin/hive-webserver \
        https://github.com/YOUR_ORG/YOUR_REPO/releases/download/v1.0.0/hive-webserver-linux-amd64
      chmod +x /shared/bin/hive-webserver
    volumeMounts:
    - name: shared-bin
      mountPath: /shared/bin

  containers:
  - name: sandbox
    image: user-specified-base-image
    command: ["/shared/bin/hive-webserver"]
    env:
    - name: PORT
      value: "8080"
    volumeMounts:
    - name: shared-bin
      mountPath: /shared/bin

  volumes:
  - name: shared-bin
    emptyDir: {}
```

### Using Init Container with Container Registry

```yaml
initContainers:
- name: inject-webserver
  image: your-registry/hive-webserver-binary:v1.0.0
  command: ["cp", "/hive-webserver", "/shared/bin/"]
  volumeMounts:
  - name: shared-bin
    mountPath: /shared/bin
```

### Enabling Binary Mode

Add annotation to Experiment resource:

```yaml
apiVersion: core.hiverge.ai/v1alpha1
kind: Experiment
metadata:
  name: my-experiment
  annotations:
    hiverge.ai/use-binary-webserver: "true"
spec:
  sandbox:
    baseImage: python:3.11-slim
    # ... rest of spec
```

## File Naming Convention

Binary filenames include the architecture to support potential multi-platform builds:

```
hive-webserver-{os}-{arch}
```

Current: `hive-webserver-linux-amd64`

Future (if needed):
- `hive-webserver-linux-arm64` (for ARM-based nodes)
- `hive-webserver-darwin-amd64` (for local testing on Intel Macs)
- `hive-webserver-darwin-arm64` (for local testing on Apple Silicon)

## Verification

Check binary properties:

```bash
# File type
file dist/hive-webserver-linux-amd64
# Output: ELF 64-bit LSB executable, x86-64

# Size
du -h dist/hive-webserver-linux-amd64
# Output: ~12M

# Verify checksum
sha256sum -c dist/hive-webserver-linux-amd64.sha256
```

Test locally:

```bash
# Run binary in container
docker run --rm \
  -e PORT=8080 \
  -v $PWD/dist:/bin \
  -p 8080:8080 \
  python:3.11-slim /bin/hive-webserver-linux-amd64
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
chmod +x /path/to/hive-webserver-linux-amd64
```

### "No such file or directory" on execution

The base image might be missing required libraries. Check with:
```bash
ldd hive-webserver-linux-amd64
```

Most Linux images include required libraries (glibc), but very minimal images like `FROM scratch` won't work.

### Download fails in init container

- Verify the download URL is accessible
- For private GitHub releases, you need a GitHub token
- Check network policies in Kubernetes cluster
- Verify DNS resolution inside cluster

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
