#!/usr/bin/env bash


if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda was not found." >&2
    echo "Download and install Conda from https://docs.conda.io/projects/miniconda/en/latest/" >&2
    echo "You might need to run conda init bash and restart your terminal after installation." >&2
    exit 1
fi

sudo apt update && sudo apt upgrade -y
sudo apt install -y portaudio19-dev libsox-dev ffmpeg

conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU installation (choose your CUDA version: cu126, cu128, cu129)
pip install -e .[cu129]

# CPU-only installation
# pip install -e cpu

# Default installation (uses PyTorch default index)
# pip install -e .

# If you encounter an error during installation due to pyaudio, consider using the following command:
# "$CONDA_CMD" install pyaudio
# Then run pip install -e . again
