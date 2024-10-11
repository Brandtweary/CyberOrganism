# Setup Instructions for CyberOrganism

## 1. Clone the Repository

To get started, clone the repository from GitHub:

```bash
git clone https://github.com/Brandtweary/CyberOrganism.git
cd CyberOrganism
```

## 2. Install Python and PyPy

Before proceeding, ensure you have Python installed on your system. To create virtual environments, you need Python 3.3 or newer. You can check your Python version by running `python --version` in your command line. If you don't have Python installed, download and install it from the official Python website (https://www.python.org/downloads/). 

When installing Python, make sure to check the option to add Python to your system PATH. This allows you to run Python from any directory in the command line. If you've already installed Python and it's not in your PATH, you can modify your system's environment variables to include the Python installation directory. If you're unsure how to do this, follow the instructions below for adding PyPy to your PATH but replace the PyPy directory with the Python directory.

For the simulation environment you'll need PyPy, which you can download from the official PyPy website (https://www.pypy.org/download.html). Follow these steps to install and configure PyPy:

1. Download PyPy3.10 from the official website: https://www.pypy.org/download.html

2. Extract the downloaded zip file to a permanent location:
   - Windows example: `C:\Program Files\pypy3.9-v7.3.9-win64`
   - Unix example: `/usr/local/pypy3.9-v7.3.9-x86_64`

3. Add PyPy to your system PATH:
   - Windows:
     1. Open "Edit the system environment variables"
     2. Click "Environment Variables"
     3. Under "System variables", edit "Path"
     4. Add the path to your PyPy directory
     5. Note: If the system path does not update immediately, restart your computer
   - Unix:
     1. Edit your shell configuration file (e.g., `~/.bashrc`)
     2. Add: `export PATH="/path/to/pypy:$PATH"`
     3. Run: `source ~/.bashrc`

4. Verify the installation:
   - Open a new terminal/command prompt
   - Run: `pypy3 --version`


## 3. Set Up Virtual Environments

First, create the necessary directories:

```bash
mkdir -p environments/main_env environments/gui_env environments/learner_env environments/simulation_env
```

Now, follow the instructions for your operating system:

### For Windows:

```bash
# Main environment
python -m venv environments\main_env
environments\main_env\Scripts\activate
pip install -r requirements.txt
deactivate

# GUI environment
python -m venv environments\gui_env
environments\gui_env\Scripts\activate
pip install -r gui\requirements.txt
deactivate

# Learner environment
python -m venv environments\learner_env
environments\learner_env\Scripts\activate
pip install -r learner\requirements.txt
deactivate

# Simulation environment
pypy3 -m venv environments\simulation_env
environments\simulation_env\Scripts\activate
pip install -r simulation\requirements.txt
deactivate
```

### For Unix-based systems (Linux, macOS):

```bash
# Main environment
python3 -m venv environments/main_env
source environments/main_env/bin/activate
pip install -r requirements.txt
deactivate

# GUI environment
python3 -m venv environments/gui_env
source environments/gui_env/bin/activate
pip install -r gui/requirements.txt
deactivate

# Learner environment
python3 -m venv environments/learner_env
source environments/learner_env/bin/activate
pip install -r learner/requirements.txt
deactivate

# Simulation environment
pypy3 -m venv environments/simulation_env
source environments/simulation_env/bin/activate
pip install -r simulation/requirements.txt
deactivate
```

Note: These instructions assume you're using `pip` as your package manager. If you're using a different package manager, adjust the install commands accordingly.

## 3. Running the Application

Once all environments are set up and dependencies are installed, you can run the application using:

```bash
python main_runner.py
```

This script will handle initializing all the subprocesses using the correct Python interpreters for each environment.

## Notes

- Ensure that the naming of the environment folders matches exactly what's specified in the `main_runner.py` file.
- You may need to adjust the paths in `main_runner.py` if your system uses different paths for Python interpreters.

## Troubleshooting

If you encounter any issues during setup or execution, please check the following:

1. Ensure all required dependencies are installed in each environment.
2. Verify that the paths to the Python interpreters in `main_runner.py` are correct for your system.
3. Make sure you have the necessary permissions to create and modify files in the project directory.

For more detailed information or if you encounter any specific errors, please refer to the project's issue tracker on GitHub.