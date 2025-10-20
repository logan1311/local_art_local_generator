#!/bin/bash
echo "🎯 Installing AI Art Generator..."
echo "=================================="

# Create virtual environment
python3 -m venv ai_env
source ai_env/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test installation
echo "🔧 Testing installation..."
python src/simple_sd.py

echo ""
echo "🎉 Installation completed!"
echo "🚀 Run: source ai_env/bin/activate && python run.py"