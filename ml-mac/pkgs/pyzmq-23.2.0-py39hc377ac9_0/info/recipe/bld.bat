set DISTUTILS_USE_SDK=1

set ZMQ_PREFIX=%LIBRARY_PREFIX%

"%PYTHON%" -m pip install .
if errorlevel 1 exit 1
