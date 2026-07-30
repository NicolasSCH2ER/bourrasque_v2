@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
cl /EHsc /O2 /std:c++17 /I "C:\Users\nicol\Code\bourrasque_v2\core\include" test_colliders.cpp /link /LIBPATH:"C:\Users\nicol\Code\bourrasque_v2\build\Release" bourrasque.lib /OUT:test_colliders.exe
cl /EHsc /O2 /std:c++17 /I "C:\Users\nicol\Code\bourrasque_v2\core\include" debug1.cpp /link /LIBPATH:"C:\Users\nicol\Code\bourrasque_v2\build\Release" bourrasque.lib /OUT:debug1.exe
cl /EHsc /O2 /std:c++17 /I "C:\Users\nicol\Code\bourrasque_v2\core\include" debug_sign.cpp /link /LIBPATH:"C:\Users\nicol\Code\bourrasque_v2\build\Release" bourrasque.lib /OUT:debug_sign.exe
