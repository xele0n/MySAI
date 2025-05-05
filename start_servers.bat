@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting the AI servers...
start cmd /k python server.py
start cmd /k python chat_server.py

echo Waiting for servers to initialize...
timeout /t 5

echo Starting the frontend...
start "" http://localhost:3000
call npm run dev 