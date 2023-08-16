

set -ex



python -c "import ctypes; ctypes.cdll['${PREFIX}/lib/libopenblas${SHLIB_EXT}']"
exit 0
