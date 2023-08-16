

set -ex



touch checksum.txt
$PREFIX/bin/openssl sha256 checksum.txt
pkg-config --print-errors --exact-version "1.1.1t" openssl
exit 0
