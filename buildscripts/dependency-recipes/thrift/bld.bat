
if %PY_VER% == 3.3 (
    xcopy /E %RECIPE_DIR%\vc10 .
)
if %PY_VER% == 3.4 (
    xcopy /E %RECIPE_DIR%\vc10 .
)
if %PY_VER% == 2.6 (
    xcopy /E %RECIPE_DIR%\vc9 .
)
if %PY_VER% == 2.7 (
    xcopy /E %RECIPE_DIR%\vc9 .
)
if errorlevel 1 exit 1

copy %RECIPE_DIR%\stdint.h .
if errorlevel 1 exit 1

mkdir %LIBRARY_INC%\thrift
mkdir %LIBRARY_INC%\thrift\protocol
mkdir %LIBRARY_INC%\thrift\processor
mkdir %LIBRARY_INC%\thrift\transport
mkdir %LIBRARY_INC%\thrift\windows
mkdir %LIBRARY_INC%\thrift\concurrency
copy lib\cpp\src\thrift\*.h %LIBRARY_INC%\thrift
copy lib\cpp\src\thrift\transport\*.h %LIBRARY_INC%\thrift\transport
copy lib\cpp\src\thrift\protocol\*.h %LIBRARY_INC%\thrift\protocol
copy lib\cpp\src\thrift\protocol\*.tcc %LIBRARY_INC%\thrift\protocol
copy lib\cpp\src\thrift\processor\*.h %LIBRARY_INC%\thrift\processor
copy lib\cpp\src\thrift\windows\*.h %LIBRARY_INC%\thrift\windows
copy lib\cpp\src\thrift\concurrency\*.h %LIBRARY_INC%\thrift\concurrency
if errorlevel 1 exit 1

if %PY_VER% == 3.3 (
    msbuild Thrift.sln /property:Configuration=Release /property:Platform=x64 
)
if %PY_VER% == 3.4 (
    msbuild Thrift.sln /property:Configuration=Release /property:Platform=x64 
)
if %PY_VER% == 2.6 (
    vcbuild Thrift.sln
)
if %PY_VER% == 2.7 (
    vcbuild Thrift.sln
)
if errorlevel 1 exit 1

if %PY_VER% == 3.3 (
    copy x64\Release\Thrift.lib %LIBRARY_LIB%
)
if %PY_VER% == 3.4 (
    copy x64\Release\Thrift.lib %LIBRARY_LIB%
)
if %PY_VER% == 2.6 (
    copy Release\Thrift.lib %LIBRARY_LIB%
)
if %PY_VER% == 2.7 (
    copy Release\Thrift.lib %LIBRARY_LIB%
)
if errorlevel 1 exit 1
