@echo off
REM Kintrace setup for Windows. Double-click or run: setup.bat
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || goto :err
python -m pip install pytest
python -m kintrace incident --outdir incidents_check || goto :err
echo.
echo Kintrace is set up. Next time, run: .venv\Scripts\activate
pause
exit /b 0
:err
echo.
echo Something failed. Copy the error above and send it to Claude.
pause
exit /b 1
