#!/bin/bash
set -e

# Build and release daemon-server binary to GitHub Releases
# This makes the binary publicly accessible via HTTPS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
VERSION=${1:-}
SKIP_BUILD=${SKIP_BUILD:-false}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0.0"
    exit 1
fi

echo "=========================================="
echo "Building and Releasing Daemon Server Binary"
echo "Version: $VERSION"
echo "=========================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "ERROR: GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "ERROR: Not authenticated with GitHub."
    echo "Run: gh auth login"
    exit 1
fi

# Architectures to build and release.
ARCHES=(amd64 arm64)

# Build binaries (unless skipped)
if [ "$SKIP_BUILD" != "true" ]; then
    for arch in "${ARCHES[@]}"; do
        echo ""
        echo "Step 1: Building binary ($arch)..."
        ./build-binary.sh "$arch"
    done
else
    echo ""
    echo "Step 1: Skipping build (SKIP_BUILD=true)"
fi

# Verify every arch asset exists and generate per-asset checksums.
echo ""
echo "Step 2: Verifying assets and generating checksums..."
ASSETS=()
cd dist
for arch in "${ARCHES[@]}"; do
    asset="daemon-server-linux-${arch}"
    if [ ! -f "$asset" ]; then
        echo "ERROR: asset not found at dist/${asset}"
        echo "Run build-binary.sh ${arch} first or set SKIP_BUILD=false"
        exit 1
    fi
    sha256sum "$asset" > "${asset}.sha256"
    sha=$(cut -d' ' -f1 < "${asset}.sha256")
    size=$(du -h "$asset" | cut -f1)
    echo "  ${asset}: size=${size} sha256=${sha}"
    ASSETS+=("$asset" "${asset}.sha256")
done
cd ..

# Check if release already exists
echo ""
echo "Step 3: Checking if release exists..."
if gh release view "$VERSION" &> /dev/null; then
    echo "  Release $VERSION already exists."
    read -p "  Delete and recreate? (y/N): " DELETE
    if [ "$DELETE" = "y" ]; then
        echo "  Deleting existing release..."
        gh release delete "$VERSION" --yes
    else
        echo "  Uploading to existing release..."
        gh release upload "$VERSION" \
            "${ASSETS[@]/#/dist/}" \
            --clobber

        echo ""
        echo "=========================================="
        echo "SUCCESS! Binaries updated in release"
        echo "=========================================="
        REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
        for arch in "${ARCHES[@]}"; do
            echo "  https://github.com/${REPO}/releases/download/${VERSION}/daemon-server-linux-${arch}"
        done
        exit 0
    fi
fi

# Create release
echo ""
echo "Step 4: Creating GitHub release..."
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

# Assemble per-arch download instructions for the release notes.
DL_DOC=""
for arch in "${ARCHES[@]}"; do
    asset="daemon-server-linux-${arch}"
    DL_DOC+="# ${arch}
curl -L -o daemon-server \\
  https://github.com/${REPO}/releases/download/${VERSION}/${asset}
curl -L -o daemon-server.sha256 \\
  https://github.com/${REPO}/releases/download/${VERSION}/${asset}.sha256
sha256sum -c daemon-server.sha256
chmod +x daemon-server

"
done

gh release create "$VERSION" \
    "${ASSETS[@]/#/dist/}" \
    --title "Daemon Server Binary $VERSION" \
    --notes "**Daemon Server Binaries for Linux (${ARCHES[*]})**

Each architecture is published as \`daemon-server-linux-<arch>\` with a matching
\`.sha256\`. The hive-operator downloads the asset matching the sandbox's arch.

## Download

\`\`\`bash
${DL_DOC}\`\`\`

## Usage

\`\`\`bash
# Run daemon server with defaults (/app and /.backup directories)
PORT=8080 ./daemon-server

# Or specify custom directories
PORT=8080 REPO_DIR=/custom/app BACKUP_DIR=/custom/backup ./daemon-server
\`\`\`
"

echo ""
echo "=========================================="
echo "SUCCESS! Release created"
echo "=========================================="
echo "View release: https://github.com/${REPO}/releases/tag/$VERSION"
echo ""
echo "Download URLs:"
for arch in "${ARCHES[@]}"; do
    echo "  https://github.com/${REPO}/releases/download/${VERSION}/daemon-server-linux-${arch}"
done
