#!/usr/bin/env python3
"""
Smart Attendance System Interface Launcher
Launches both the web interface and Streamlit dashboard
"""

import os
import sys
import webbrowser
import subprocess
import time
import threading
from pathlib import Path

def open_web_interface():
    """Open the web interface in the default browser"""
    try:
        web_file = Path(__file__).parent / "web_interface.html"
        if web_file.exists():
            webbrowser.open(f"file://{web_file.absolute()}")
            print("✅ Web interface opened in browser")
        else:
            print("❌ Web interface file not found")
    except Exception as e:
        print(f"❌ Error opening web interface: {e}")

def launch_streamlit():
    """Launch the Streamlit dashboard"""
    try:
        print("🚀 Starting Streamlit dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501", "--server.headless", "true"])
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("📹 Smart Attendance System Interface Launcher")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ["dashboard.py", "face_recognition_system.py", "database.py"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all system files are present.")
        return
    
    print("✅ All required files found")
    
    # Launch options
    print("\n🎯 Choose an interface option:")
    print("1. Web Interface (HTML) - Quick overview")
    print("2. Streamlit Dashboard - Full functionality")
    print("3. Both - Web interface + Streamlit dashboard")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🌐 Opening web interface...")
                open_web_interface()
                break
                
            elif choice == "2":
                print("\n🚀 Launching Streamlit dashboard...")
                print("The dashboard will open at: http://localhost:8501")
                print("Press Ctrl+C to stop the server")
                launch_streamlit()
                break
                
            elif choice == "3":
                print("\n🚀 Launching both interfaces...")
                
                # Start Streamlit in a separate thread
                streamlit_thread = threading.Thread(target=launch_streamlit, daemon=True)
                streamlit_thread.start()
                
                # Wait a moment for Streamlit to start
                print("⏳ Starting Streamlit server...")
                time.sleep(3)
                
                # Open web interface
                print("🌐 Opening web interface...")
                open_web_interface()
                
                print("\n✅ Both interfaces launched!")
                print("📊 Streamlit Dashboard: http://localhost:8501")
                print("🌐 Web Interface: Opened in browser")
                print("\nPress Ctrl+C to stop all services")
                
                try:
                    # Keep the main thread alive
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down...")
                    break
                    
            elif choice == "4":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
