import os
import sys

# ---------------------------------------------
# PROJECT ROOT
# ---------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.spectral_subtraction import reduce_noise


# ---------------------------------------------
# FILE PATHS
# ---------------------------------------------

input_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "noisy",
    "noisy_voice1.wav"
)

output_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "enhanced",
    "test_noise_reduced_voice1.wav"
)


# ---------------------------------------------
# CHECK INPUT
# ---------------------------------------------

print("\n================================")
print("     SONIC SHIELD")
print("     NOISE REDUCTION TEST")
print("================================\n")

print("Input:")
print(input_file)

print("\nOutput:")
print(output_file)


if not os.path.exists(input_file):

    print("\nERROR: Input audio file not found.")
    print(input_file)
    sys.exit(1)


# ---------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)


# ---------------------------------------------
# RUN NOISE REDUCTION
# ---------------------------------------------

print("\nStarting noise reduction...\n")

try:

    result = reduce_noise(
        input_file,
        output_file
    )

    print("\n================================")
    print("✓ NOISE REDUCTION COMPLETED")
    print("================================")

    print("\nOutput file:")
    print(output_file)

    if os.path.exists(output_file):

        print("\n✓ Enhanced WAV file created successfully.")

    else:

        print("\n⚠ Function completed but output file was not found.")

except Exception as e:

    print("\n================================")
    print("ERROR")
    print("================================")

    print(type(e).__name__)
    print(str(e))