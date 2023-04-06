

set -ex



python -m unittest tests/test_zipp.py
pip check
exit 0
