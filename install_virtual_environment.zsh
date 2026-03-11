echo "Creating a virtual environment for Python"
python -m venv venv
echo "Activating the virutal environment"
source venv/bin/activate
pip install --upgrade pip &&
pip install -r .\requirements.txt
