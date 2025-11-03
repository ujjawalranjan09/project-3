#!/bin/bash
# Quick Start Script for Railway Traffic Control System

echo "========================================"
echo "Railway Traffic Control System"
echo "Quick Start Installation"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for dataset
echo ""
if [ -f "railway_OPTIMIZED_BALANCED_500K.csv" ]; then
    echo "✓ Dataset found"

    # Train models
    echo ""
    echo "Training ML models..."
    echo "(This may take 5-10 minutes)"
    python train_ml_models.py

    echo ""
    echo "✓ Models trained successfully"
else
    echo "⚠ Dataset not found: railway_OPTIMIZED_BALANCED_500K.csv"
    echo "Please place your dataset in the root directory and run:"
    echo "  python train_ml_models.py"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p backend/models
mkdir -p logs
mkdir -p data

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To start the system:"
echo "  1. Backend API:  python flask_app.py"
echo "  2. Frontend:     Open dashboard.html in browser"
echo ""
echo "API will be available at: http://localhost:5000"
echo ""
echo "For detailed instructions, see README.md"
echo "========================================"
