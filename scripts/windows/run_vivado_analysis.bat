@echo off
REM Run synthesis + implementation + timing/utilization reports.
REM Requires Vivado on PATH (use the Vivado Command Prompt).

setlocal
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"

echo ================================================================
echo  Running Vivado analysis from:
echo    %REPO_ROOT%
echo ================================================================

where vivado >nul 2>&1
if errorlevel 1 (
  echo ERROR: vivado not found on PATH.
  echo Open the "Vivado Tcl Shell" / "Xilinx Vivado Command Prompt" and retry.
  popd
  exit /b 1
)

vivado -mode batch -source "%REPO_ROOT%\scripts\run_vivado_analysis.tcl"
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
