



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
pip list
IF %ERRORLEVEL% NEQ 0 exit /B 1
pip list | grep -iE "terminado\s*0\.17\.1"
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
