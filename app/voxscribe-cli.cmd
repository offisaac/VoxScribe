@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "HF_HOME=%ROOT%\models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%ROOT%\models\huggingface\hub"
set "TORCH_HOME=%ROOT%\models\torch"
"%ROOT%\runtime\Scripts\python.exe" "%ROOT%\app\voxscribe_cli.py" %*
exit /b %errorlevel%
