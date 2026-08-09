@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
cl /EHsc /O2 /std:c++17 /I "C:\Users\nicol\Code\bourrasque_v2\core\include" test_nonregression_a1.cpp /link /LIBPATH:"C:\Users\nicol\Code\bourrasque_v2\build\Release" bourrasque.lib /OUT:nr_new.exe
cl /EHsc /O2 /std:c++17 /I "C:\tmp\pre_a1\core\include" test_nonregression_a1.cpp /link /LIBPATH:"C:\tmp\pre_a1\build\Release" bourrasque.lib /OUT:nr_ref.exe
