#!/bin/bash

# Speech-to-Text Installation Script for Curious Agent
# This script installs the necessary dependencies for speech-to-text functionality

set -e  # Exit on any error

echo "🎤 Installing Speech-to-Text Dependencies for Curious Agent"
echo "=========================================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️ No virtual environment detected. It's recommended to use a virtual environment."
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please create and activate a virtual environment first:"
        echo "  python3 -m venv .venv"
        echo "  source .venv/bin/activate"
        exit 1
    fi
fi

# Detect operating system
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "🖥️ Detected OS: $OS"

# Install system dependencies
echo "📦 Installing system dependencies..."

if [[ "$OS" == "linux" ]]; then
    echo "Installing Linux dependencies..."
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg
    echo "✅ Linux dependencies installed"
    
elif [[ "$OS" == "macos" ]]; then
    echo "Installing macOS dependencies..."
    if command -v brew &> /dev/null; then
        brew install portaudio ffmpeg
        echo "✅ macOS dependencies installed via Homebrew"
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
elif [[ "$OS" == "windows" ]]; then
    echo "⚠️ Windows detected. Please install PyAudio manually:"
    echo "  1. Download PyAudio wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio"
    echo "  2. Install with: pip install [downloaded_wheel_file]"
    echo "  3. Or use conda: conda install pyaudio"
    read -p "Press Enter after installing PyAudio..."
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install basic dependencies
echo "Installing basic STT dependencies..."
pip install speechrecognition
pip install pydub

# Install PyAudio (this might fail on some systems)
echo "Installing PyAudio..."
if pip install pyaudio; then
    echo "✅ PyAudio installed successfully"
else
    echo "❌ PyAudio installation failed. This is common on some systems."
    echo "Please install PyAudio manually:"
    if [[ "$OS" == "linux" ]]; then
        echo "  sudo apt-get install python3-pyaudio"
    elif [[ "$OS" == "macos" ]]; then
        echo "  brew install portaudio"
        echo "  pip install pyaudio"
    else
        echo "  Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio"
    fi
fi

# Install Whisper dependencies (optional)
echo "Installing Whisper dependencies (optional)..."
if pip install torch torchaudio; then
    echo "✅ PyTorch installed successfully"
    if pip install openai-whisper; then
        echo "✅ Whisper installed successfully"
    else
        echo "⚠️ Whisper installation failed, but basic STT will still work"
    fi
else
    echo "⚠️ PyTorch installation failed, Whisper will not be available"
fi

# Install remaining requirements
echo "Installing remaining requirements..."
pip install -r requirements.txt

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Test the installation: python test_stt.py"
echo "2. Run the Curious Agent: metta main.metta"
echo "3. Choose 'speech' or 'mixed' mode when prompted"
echo ""
echo "📚 For more information, see SPEECH_TO_TEXT_GUIDE.md"
echo ""
echo "🔧 If you encounter issues:"
echo "- Check microphone permissions"
echo "- Ensure microphone is not being used by other applications"
echo "- Test microphone with system audio settings"
echo "- Run the test script to diagnose problems"
