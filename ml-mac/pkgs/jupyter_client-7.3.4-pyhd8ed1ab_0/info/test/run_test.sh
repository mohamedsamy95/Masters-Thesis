

set -ex



pip check
jupyter kernelspec list
jupyter run -h
pytest --pyargs jupyter_client --cov jupyter_client --cov-report term-missing:skip-covered --no-cov-on-fail -k "not test_signal_kernel_subprocesses"
exit 0
