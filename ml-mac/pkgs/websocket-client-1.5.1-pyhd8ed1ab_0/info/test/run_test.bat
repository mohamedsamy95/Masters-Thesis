



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
wsdump --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0