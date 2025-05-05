import requests
import sys

def check_server(url):
    try:
        response = requests.get(url)
        print(f"Server at {url} is running!")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"Failed to connect to {url}")
        print(f"Error: {str(e)}")
        return False

def main():
    print("Checking server status...")
    
    simulation_running = check_server("http://localhost:5000/health")
    print("\n")
    
    chat_running = check_server("http://localhost:5001/health")
    print("\n")
    
    frontend_running = check_server("http://localhost:3000")
    
    if simulation_running and chat_running and frontend_running:
        print("\nAll servers are running!")
    else:
        print("\nSome servers are not running.")
        print("To start servers, run: start_servers.bat (Windows) or start_servers.sh (Mac/Linux)")

if __name__ == "__main__":
    main() 