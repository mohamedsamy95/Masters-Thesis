



pip check
IF %ERRORLEVEL% NEQ 0 exit /B 1
jsonschema --version
IF %ERRORLEVEL% NEQ 0 exit /B 1
jsonschema --help
IF %ERRORLEVEL% NEQ 0 exit /B 1
jsonschema --version | grep 4\.17\.3
IF %ERRORLEVEL% NEQ 0 exit /B 1
exit /B 0
