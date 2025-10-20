#!/usr/bin/env python3
"""
🎨 AI Art Generator - Cross Platform Launcher
"""
import os
import sys
import platform

def main():
    print("🎨 AI Art Generator")
    print("==================")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from simple_sd import main as sd_main
        sd_main()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you installed dependencies:")
        print("   pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()