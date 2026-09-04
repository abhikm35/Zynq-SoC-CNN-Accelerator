@echo off
REM Open the generated Vivado project in the GUI.
REM Create the project first if it does not exist.

setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
set "XPR=%REPO_ROOT%\vivado_build\cnn_accelerator\cnn_accelerator.xpr"

where vivado >nul 2>&1
if errorlevel 1 (
  echo ERROR: vivado not found on PATH.
  echo Open the "Vivado Tcl Shell" / "Xilinx Vivado Command Prompt" and retry.
  popd
  exit /b 1
)

if not exist "%XPR%" (
  echo Project not found:
  echo   %XPR%
  echo Creating it now...
  call "%SCRIPT_DIR%create_vivado_project.bat"
  if errorlevel 1 (
    popd
    exit /b 1
  )
)

echo Opening %XPR%
start "" vivado "%XPR%"
popd
exit /b 0
