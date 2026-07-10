#!/usr/bin/env bash
set -euo pipefail

echo "🔨 Building Elpis release binary..."
cd "$(dirname "$0")/tui"
cargo build --release

echo "📁 Creating Debian package directory structure..."
BUILD_DIR="/tmp/elpis_deb_build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/usr/bin"
mkdir -p "${BUILD_DIR}/DEBIAN"

echo "💾 Copying binary..."
cp target/release/elpis "${BUILD_DIR}/usr/bin/"

echo "📄 Creating control file..."
cat << 'EOF' > "${BUILD_DIR}/DEBIAN/control"
Package: elpis
Version: 0.1.0
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Masih Moafi <masihmoafi@gmail.com>
Description: Project Elpis TUI client
EOF

echo "📦 Packaging .deb..."
cd ..
dpkg-deb --build "${BUILD_DIR}" elpis_0.1.0_amd64.deb

echo "✅ Debian package created successfully: elpis_0.1.0_amd64.deb"
echo "👉 Install it globally with: sudo dpkg -i elpis_0.1.0_amd64.deb"
