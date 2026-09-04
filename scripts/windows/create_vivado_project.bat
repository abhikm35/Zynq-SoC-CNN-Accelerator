@echo off
REM Create the Vivado project from repository RTL (batch mode).
REM Run from a Vivado Command Prompt (or any shell where "vivado" is on PATH).
REM Do not hardcode the Vivado install directory.

setlocal
set "SCRIPT_DIR=%~dp0"
REM scripts\windows -> repo root
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"

echo ================================================================
echo  Creating Vivado project from:
echo    %REPO_ROOT%
echo ================================================================

where vivado >nul 2>&1
if errorlevel 1 (
  echo ERROR: vivado not found on PATH.
  echo Open the "Vivado Tcl Shell" / "Xilinx Vivado Command Prompt" and retry.
  popd
  exit /b 1
)

vivado -mode batch -source "%REPO_ROOT%\scripts\create_vivado_project.tcl"
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
