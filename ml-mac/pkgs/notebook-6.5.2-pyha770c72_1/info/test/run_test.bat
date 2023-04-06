



python -m pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
jupyter notebook -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
jupyter nbextension -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
jupyter serverextension -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
jupyter bundlerextension -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
