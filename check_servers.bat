@echo off
echo Checking server status...

echo.
echo Checking simulation server (http://localhost:5000)...
curl -s http://localhost:5000/health
IF %ERRORLEVEL% NEQ 0 (
    echo FAILED: Simulation server is not running
) ELSE (
    echo SUCCESS: Simulation server is running
)

echo.
echo Checking chat server (http://localhost:5001)...
curl -s http://localhost:5001/health
IF %ERRORLEVEL% NEQ 0 (
    echo FAILED: Chat server is not running
) ELSE (
    echo SUCCESS: Chat server is running
)

echo.
echo Checking frontend server (http://localhost:3000)...
curl -s http://localhost:3000
IF %ERRORLEVEL% NEQ 0 (
    echo FAILED: Frontend server is not running
) ELSE (
    echo SUCCESS: Frontend server is running
)

echo.
echo If any server failed, make sure to run start_servers.bat
echo For manual startup:
echo 1. python server.py
echo 2. python chat_server.py
echo 3. npm run dev

pause 