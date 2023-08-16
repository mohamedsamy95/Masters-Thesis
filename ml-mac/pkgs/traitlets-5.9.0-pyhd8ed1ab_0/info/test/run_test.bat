



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
pytest -vv --pyargs traitlets --cov traitlets --no-cov-on-fail --cov-report term-missing:skip-covered --cov-fail-under=92
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
