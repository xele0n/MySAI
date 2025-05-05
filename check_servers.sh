#!/bin/bash

echo "Checking server status..."

echo
echo "Checking simulation server (http://localhost:5000)..."
if curl -s http://localhost:5000/health > /dev/null; then
    echo "SUCCESS: Simulation server is running"
else
    echo "FAILED: Simulation server is not running"
fi

echo
echo "Checking chat server (http://localhost:5001)..."
if curl -s http://localhost:5001/health > /dev/null; then
    echo "SUCCESS: Chat server is running"
else
    echo "FAILED: Chat server is not running"
fi

echo
echo "Checking frontend server (http://localhost:3000)..."
if curl -s http://localhost:3000 > /dev/null; then
    echo "SUCCESS: Frontend server is running"
else
    echo "FAILED: Frontend server is not running"
fi

echo
echo "If any server failed, make sure to run start_servers.sh"
echo "For manual startup:"
echo "1. python server.py"
echo "2. python chat_server.py"
echo "3. npm run dev"

read -p "Press any key to continue..." -n1 -s 