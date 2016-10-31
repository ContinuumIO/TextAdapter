cd src\interfaces\libpq
copy %RECIPE_DIR%\win32.mak .
if errorlevel 1 exit 1

nmake /f win32.mak CPU=AMD64
if errorlevel 1 exit 1

copy libpq-fe.h %LIBRARY_INC%
copy ..\..\include\postgres_ext.h %LIBRARY_INC%
copy Release\libpq.lib %LIBRARY_LIB%
if errorlevel 1 exit 1
