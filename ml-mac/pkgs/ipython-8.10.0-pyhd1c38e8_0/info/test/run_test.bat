



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
pygmentize -L | grep ipython
IF %ERRORLEVEL% NEQ 0 exit /B 1
ipython -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
ipython3 -h
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
