

set -ex



test -f ${PREFIX}/include/sodium.h
test -f ${PREFIX}/lib/libsodium.a
test -f ${PREFIX}/lib/libsodium.dylib
exit 0
