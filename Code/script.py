import subprocess

# Define commands to execute
commands = [
    "pip install datasets==1.0.2",
    "pip install tqdm==4.57.0",
    "pip install scikit-learn",
    "python data_preprocess.py",
    "pip install --quiet transformers==4.28.1",
    "conda install -c conda-forge sentencepiece",
    "pip install torch",
    "pip install pytorch-lightning",
    "pip install gdown"
]

# Execute commands
for cmd in commands:
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(e)
