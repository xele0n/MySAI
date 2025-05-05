import subprocess
import time
import sys
import os
import webbrowser
import threading

def run_command(command):
    try:
        process = subprocess.Popen(command, shell=True)
        return process
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return None

def check_server(url):
    import requests
    try:
        response = requests.get(url, timeout=2)
        return True
    except:
        return False

def main():
    print("Starting MySAI application...")
    
    # Start the simulation server
    print("Starting simulation server...")
    sim_process = run_command("python server.py")
    
    # Start the chat server
    print("Starting chat server...")
    chat_process = run_command("python chat_server.py")
    
    # Wait for servers to initialize
    print("Waiting for servers to initialize...")
    servers_ready = False
    timeout = 20  # 20 seconds timeout
    start_time = time.time()
    
    while not servers_ready and (time.time() - start_time) < timeout:
        sim_ready = check_server("http://localhost:5000/health")
        chat_ready = check_server("http://localhost:5001/health")
        
        if sim_ready and chat_ready:
            servers_ready = True
            break
        
        print(".", end="", flush=True)
        time.sleep(1)
    
    print("\n")
    
    if servers_ready:
        print("Servers initialized successfully!")
        
        # Start the frontend
        print("Starting frontend...")
        if sys.platform.startswith('win'):
            frontend_process = run_command("npm run dev")
        else:
            frontend_process = run_command("npm run dev")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(5)
            webbrowser.open("http://localhost:3000")
            
        threading.Thread(target=open_browser).start()
        
        print("Application started!")
        print("Visit http://localhost:3000 in your browser")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down servers...")
            if sim_process:
                sim_process.terminate()
            if chat_process:
                chat_process.terminate()
            if frontend_process:
                frontend_process.terminate()
    else:
        print("Failed to initialize servers within timeout period.")
        print("Please check for errors in the server logs.")
        
        if sim_process:
            sim_process.terminate()
        if chat_process:
            chat_process.terminate()

if __name__ == "__main__":
    main() 