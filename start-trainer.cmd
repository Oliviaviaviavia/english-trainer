@echo off
cd /d "%~dp0"
start "" http://localhost:8000
"%~dp0.venv\Scripts\python.exe" -X utf8 -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
