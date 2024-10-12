# Setup Instructions for CyberOrganism

## 1. Clone the Repository

To get started, clone the repository from GitHub:

```bash
git clone https://github.com/Brandtweary/CyberOrganism.git
cd CyberOrganism
```

## 2. Install Python

Ensure you have Python installed on your system. You can check your Python version by running `python --version` in your command line. If you don't have Python installed, download and install it from the official Python website (https://www.python.org/downloads/). 

When installing Python, make sure to check the option to add Python to your system PATH. This allows you to run Python from any directory in the command line.

## 3. Install Dependencies

Install the required packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## 4. Install PyTorch

PyTorch installation depends on your system's CUDA version (if you have an NVIDIA GPU) or CPU-only if you don't have a compatible GPU.

### Determine CUDA Version (for NVIDIA GPU users)

If you have an NVIDIA GPU, you need to determine your CUDA version:

#### For Windows:
1. Open Command Prompt
2. Run: `nvidia-smi`

#### For Linux:
1. Open Terminal
2. Run: `nvidia-smi`

Look for the CUDA Version in the top-right corner of the output.

### Install PyTorch

Visit the official PyTorch website (https://pytorch.org/get-started/locally/) and use the installation selector to get the correct command for your system.

#### Example for CUDA 11.7:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### For CPU-only systems:
```bash
pip install torch torchvision torchaudio
```

Replace the above commands with the one generated on the PyTorch website for your specific system configuration.

## 5. Running the Application

Once all dependencies are installed, you can run the application using:

```bash
python main_runner.py
```

## Notes

- Ensure that all required dependencies are installed in your Python environment.
-The simulation will run optimally if your system has 4 or more CPU cores.
