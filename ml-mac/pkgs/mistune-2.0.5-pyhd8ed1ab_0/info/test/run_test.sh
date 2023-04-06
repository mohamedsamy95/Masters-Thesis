

set -ex



pip check
pytest --cov mistune --cov-report=term-missing:skip-covered --cov-fail-under=98 --no-cov-on-fail
exit 0
