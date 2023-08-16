



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
jupyter kernelspec list
IF %ERRORLEVEL% NEQ 0 exit /B 1
jupyter run -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
pytest --pyargs jupyter_client --cov jupyter_client --cov-report term-missing:skip-covered --no-cov-on-fail -k "not test_signal_kernel_subprocesses"
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
