@echo off
setlocal
echo ==============================================================================
echo Launching InspectAI Industrial Inspection Application...
echo ==============================================================================

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" run.py %*
endlocal

