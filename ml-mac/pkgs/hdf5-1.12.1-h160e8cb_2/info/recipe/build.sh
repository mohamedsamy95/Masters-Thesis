#!/bin/bash

# Get an updated config.sub and config.guess
cp -r ${BUILD_PREFIX}/share/libtool/build-aux/config.* ./config
cp -r ${BUILD_PREFIX}/share/libtool/build-aux/config.* ./bin

export LIBRARY_PATH="${PREFIX}/lib"

export CC=$(basename ${CC})
export CXX=$(basename ${CXX})
export F95=$(basename ${F95})
export FC=$(basename ${FC})
export GFORTRAN=$(basename ${GFORTRAN})

if [ $(uname -s) = "Linux" ] && [ ! -f "${BUILD_PREFIX}/bin/strings" ]; then
    ln -s "${BUILD}-strings" "${BUILD_PREFIX}/bin/strings"
fi

./configure --prefix="${PREFIX}" \
            --host="${HOST}" \
            --build="${BUILD}" \
            --enable-linux-lfs \
            --with-zlib="${PREFIX}" \
            --with-pthread=yes  \
            --enable-cxx \
            --enable-fortran \
            --enable-fortran2003 \
            --with-default-plugindir="${PREFIX}/lib/hdf5/plugin" \
            --with-default-api-version=v18 \
            --enable-threadsafe \
            --enable-build-mode=production \
            --enable-unsupported \
            --enable-using-memchecker \
            --enable-clear-file-buffers \
            --enable-ros3-vfd \
            --with-ssl

make -j "${CPU_COUNT}" ${VERBOSE_AT}
if [[ ! ${HOST} =~ .*powerpc64le.* ]]; then
  # https://github.com/h5py/h5py/issues/817
  # https://forum.hdfgroup.org/t/hdf5-1-10-long-double-conversions-tests-failed-in-ppc64le/4077
  make check
fi
make install

rm -rf $PREFIX/share/hdf5_examples
