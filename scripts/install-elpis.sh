#!/usr/bin/env bash
set -euo pipefail

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) asset=elpis-linux-x86_64 ;;
  *)
    printf 'Elpis currently publishes a Linux x86_64 binary only.\n' >&2
    exit 1
    ;;
esac

repository=${ELPIS_GITHUB_REPOSITORY:-MasihMoafi/Elpis}
install_dir=${ELPIS_INSTALL_DIR:-"$HOME/.local/bin"}
release_url="https://github.com/$repository/releases/latest/download"
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT

curl --fail --location --silent --show-error \
  "$release_url/$asset" --output "$temporary_dir/$asset"
curl --fail --location --silent --show-error \
  "$release_url/$asset.sha256" --output "$temporary_dir/$asset.sha256"

(
  cd "$temporary_dir"
  sha256sum --check "$asset.sha256"
)

mkdir -p "$install_dir"
install -m 0755 "$temporary_dir/$asset" "$install_dir/.elpis.installing"
mv -f "$install_dir/.elpis.installing" "$install_dir/elpis"
printf 'Installed Elpis at %s\n' "$install_dir/elpis"
