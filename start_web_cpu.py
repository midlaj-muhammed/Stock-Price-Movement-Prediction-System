#!/usr/bin/env python3
"""
Start the web interface with CPU-only mode to avoid GPU issues.
"""

import os
import sys
import subprocess

# Force CPU usage before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

def main():
    """Start Streamlit with CPU-only configuration."""
    
    print("🚀 Starting Stock Price Prediction Web Interface (CPU Mode)")
    print("=" * 60)
    print("🔧 GPU disabled to avoid compilation issues")
    print("📊 Web interface will be available at: http://localhost:8501")
    print("=" * 60)
    
    try:
        # Start Streamlit with the web app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "src/web/app.py", 
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Set environment for the subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'
        env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        
        # Start the process
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n⏹️ Web interface stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start web interface: {e}")
        print("\n🔧 Try running directly:")
        print("   CUDA_VISIBLE_DEVICES=-1 streamlit run src/web/app.py")

if __name__ == "__main__":
    main()
