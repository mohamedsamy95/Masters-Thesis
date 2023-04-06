

set -ex



pip check
jupyter -h
jupyter-migrate -h
jupyter-troubleshoot --help
python -m pytest -vv --pyargs jupyter_core -k "not (test_not_on_path or test_path_priority)"
exit 0
