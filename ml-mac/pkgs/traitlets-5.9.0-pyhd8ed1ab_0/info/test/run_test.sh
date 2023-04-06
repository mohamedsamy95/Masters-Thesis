

set -ex



pip check
pytest -vv --pyargs traitlets --cov traitlets --no-cov-on-fail --cov-report term-missing:skip-covered --cov-fail-under=92
exit 0
