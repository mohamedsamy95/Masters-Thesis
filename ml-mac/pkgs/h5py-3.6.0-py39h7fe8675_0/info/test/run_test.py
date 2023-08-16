#  tests for h5py-3.6.0-py39h7fe8675_0 (this is a generated file);
print('===== testing package: h5py-3.6.0-py39h7fe8675_0 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
import os

os.environ['OMPI_MCA_plm'] = 'isolated'
os.environ['OMPI_MCA_btl_vader_single_copy_mechanism'] = 'none'
os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = 'yes'

import h5py
import h5py._conv
import h5py._errors
import h5py._objects
import h5py._proxy
import h5py.defs
import h5py.h5
import h5py.h5a
import h5py.h5d
import h5py.h5f
import h5py.h5fd
import h5py.h5g
import h5py.h5i
import h5py.h5l
import h5py.h5o
import h5py.h5p
import h5py.h5r
import h5py.h5s
import h5py.h5t
import h5py.h5z
import h5py.utils

# verify that mpi builds are built with mpi
should_have_mpi = os.getenv('mpi', 'nompi') != 'nompi'
have_mpi = h5py.get_config().mpi
assert have_mpi == should_have_mpi, "Expected mpi=%r, got %r" % (should_have_mpi, have_mpi)

from sys import exit
# we have file access issues with the ros3 test
# exit(h5py.run_tests())
h5py.run_tests()

#  --- run_test.py (end) ---

print('===== h5py-3.6.0-py39h7fe8675_0 OK =====');
print("import: 'h5py'")
import h5py

print("import: 'h5py._hl'")
import h5py._hl

print("import: 'h5py.tests'")
import h5py.tests

print("import: 'h5py.tests.data_files'")
import h5py.tests.data_files

print("import: 'h5py.tests.test_vds'")
import h5py.tests.test_vds

