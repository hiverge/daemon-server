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

# Build binary (unless skipped)
if [ "$SKIP_BUILD" != "true" ]; then
    echo ""
    echo "Step 1: Building binary..."
    ./build-binary.sh
else
    echo ""
    echo "Step 1: Skipping build (SKIP_BUILD=true)"
fi

# Check if binary exists
if [ ! -f dist/daemon-server ]; then
    echo "ERROR: Binary not found at dist/daemon-server"
    echo "Run build-binary.sh first or set SKIP_BUILD=false"
    exit 1
fi

# Generate checksum
echo ""
echo "Step 2: Generating checksum..."
cd dist
sha256sum daemon-server > daemon-server.sha256
SHA256=$(cat daemon-server.sha256 | cut -d' ' -f1)
SIZE=$(du -h daemon-server | cut -f1)
cd ..

echo "  Binary size: $SIZE"
echo "  SHA256: $SHA256"

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
            dist/daemon-server \
            dist/daemon-server.sha256 \
            --clobber

        echo ""
        echo "=========================================="
        echo "SUCCESS! Binary updated in release"
        echo "=========================================="
        echo "Download URL:"
        echo "  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$VERSION/daemon-server"
        echo ""
        echo "SHA256: $SHA256"
        exit 0
    fi
fi

# Create release
echo ""
echo "Step 4: Creating GitHub release..."
gh release create "$VERSION" \
    dist/daemon-server \
    dist/daemon-server.sha256 \
    --title "Daemon Server Binary $VERSION" \
    --notes "**Daemon Server Binary for Linux AMD64**

Size: $SIZE
SHA256: \`$SHA256\`

## Download

\`\`\`bash
# Download binary
curl -L -o daemon-server \\
  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$VERSION/daemon-server

# Verify checksum
curl -L -o daemon-server.sha256 \\
  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$VERSION/daemon-server.sha256
sha256sum -c daemon-server.sha256

# Make executable
chmod +x daemon-server
\`\`\`

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
echo "View release: https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$VERSION"
echo ""
echo "Download URL:"
echo "  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$VERSION/daemon-server"
echo ""
echo "SHA256: $SHA256"
