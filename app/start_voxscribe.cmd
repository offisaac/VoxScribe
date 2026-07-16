@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "HF_HOME=%ROOT%\models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%ROOT%\models\huggingface\hub"
set "PIP_CACHE_DIR=%ROOT%\cache\pip"
set "TEMP=%ROOT%\cache\temp"
set "TMP=%ROOT%\cache\temp"
if not exist "%TEMP%" mkdir "%TEMP%"
start "VoxScribe" "%ROOT%\runtime\Scripts\pythonw.exe" "%ROOT%\app\voxscribe.py"
endlocal
