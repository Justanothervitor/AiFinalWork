@echo off
echo "Creating a virtual environment for Python"
python -m venv venv
echo "Activating the virtual environment"
call venv\Scripts\activate.bat
echo "Upgrading pip and installing requirements"
pip install --upgrade pip
pip install -r requirements.txt
echo "Virtual environment setup complete."
