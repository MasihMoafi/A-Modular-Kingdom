#!/usr/bin/env bash
set -euo pipefail

source_binary=${1:?"usage: scripts/install-elpis-binary.sh PATH_TO_ELPIS_BINARY"}
install_dir=${ELPIS_INSTALL_DIR:-"$HOME/.local/bin"}
destination="$install_dir/elpis"
temporary="$install_dir/.elpis.installing"

test -f "$source_binary"
mkdir -p "$install_dir"
install -m 0755 "$source_binary" "$temporary"
mv -f "$temporary" "$destination"

printf 'Installed Elpis at %s\n' "$destination"
