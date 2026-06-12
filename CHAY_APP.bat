@echo off
cd /d "%~dp0"
echo Dang cai thu vien can thiet...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Cai thu vien gap loi. Hay kiem tra Python/pip roi chay lai.
  pause
  exit /b 1
)
echo Dang mo giao dien Streamlit...
python -m streamlit run app.py
pause
