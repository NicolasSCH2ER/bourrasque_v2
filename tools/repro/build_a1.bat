@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
cl /EHsc /O2 /std:c++17 /I "C:\Users\nicol\Code\bourrasque_v2\core\include" test_rigidbody_a1.cpp /link /LIBPATH:"C:\Users\nicol\Code\bourrasque_v2\build\Release" bourrasque.lib /OUT:test_rigidbody_a1.exe
