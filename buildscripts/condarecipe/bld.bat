copy %LIBRARY_LIB%\boost_thread-vc90-mt-1_60.lib %LIBRARY_LIB%\libboost_thread-vc90-mt-1_60.lib
%PYTHON% setup.py install
if errorlevel 1 exit 1
