

set -ex



pip check
jupyter server -h
cd tests && pytest -vv --cov jupyter_server --cov-report term-missing:skip-covered --no-cov-on-fail --cov-fail-under 70
exit 0
