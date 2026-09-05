import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import enhance_speech


input_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "noisy",
    "voice1.wav"
)

output_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "enhanced",
    "voice1_enhanced.wav"
)


enhance_speech(
    input_file,
    output_file
)