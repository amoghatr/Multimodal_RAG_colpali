#!/usr/bin/env python3
"""
Script to run both the FastAPI backend and Streamlit frontend
"""
import os
import signal
import subprocess
import sys
import time


def run_backend():
    """Start the FastAPI backend"""
    return subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
    )


def run_frontend():
    """Start the Streamlit frontend"""
    return subprocess.Popen(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            "frontend.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ]
    )


def main():
    print("🚀 Starting ColPali RAG Application...")
    print("📊 Backend API will be available at: http://localhost:8000")
    print("🌐 Frontend UI will be available at: http://localhost:8501")
    print("\nPress Ctrl+C to stop both services\n")

    # Start both processes
    backend_process = None
    frontend_process = None

    try:
        # Start backend
        print("Starting FastAPI backend...")
        backend_process = run_backend()
        time.sleep(3)  # Give backend time to start

        # Start frontend
        print("Starting Streamlit frontend...")
        frontend_process = run_frontend()

        print("\n✅ Both services are starting up!")
        print("🔗 Open http://localhost:8501 in your browser to use the application")

        # Wait for processes
        while True:
            if backend_process.poll() is not None:
                print("❌ Backend process ended unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process ended unexpectedly")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")

    finally:
        # Clean up processes
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
        print("✅ Services stopped")


if __name__ == "__main__":
    main()
