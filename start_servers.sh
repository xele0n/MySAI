#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Starting the AI servers..."
# Start the simulation server in the background
python server.py > simulation_server.log 2>&1 &
SIM_PID=$!
echo "Simulation server started with PID: $SIM_PID"

# Start the chat server in the background
python chat_server.py > chat_server.log 2>&1 &
CHAT_PID=$!
echo "Chat server started with PID: $CHAT_PID"

# Make script to kill the servers later
cat > stop_servers.sh << EOF
#!/bin/bash
kill $SIM_PID
kill $CHAT_PID
echo "Servers stopped"
EOF

chmod +x stop_servers.sh

echo "Waiting for servers to initialize..."
sleep 5

echo "Starting the frontend..."
# Open browser to the frontend URL (works on most Unix systems)
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:3000 &
elif command -v open > /dev/null; then
    open http://localhost:3000 &
fi

npm run dev 