



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
pytest --cov mistune --cov-report=term-missing:skip-covered --cov-fail-under=98 --no-cov-on-fail
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
