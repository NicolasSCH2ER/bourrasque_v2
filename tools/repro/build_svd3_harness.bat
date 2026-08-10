@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
nvcc -O2 -arch=native -std=c++17 svd3_harness.cu -o svd3_harness.exe
