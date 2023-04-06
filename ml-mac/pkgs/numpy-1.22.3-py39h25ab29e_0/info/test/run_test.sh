

set -ex



f2py -h
python -c "import numpy; numpy.show_config()"
export OPENBLAS_NUM_THREADS=1
pytest -vvv --pyargs numpy -k "not (_not_a_real_test or test_unary_PyUFunc_O_O_method_full[reciprocal] or test_new_policy or test_limited_api)" --durations=0
exit 0
