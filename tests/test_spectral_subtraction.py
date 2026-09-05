import os
import sys

# Get project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import function
from src.spectral_subtraction import reduce_noise


def test_spectral_subtraction():

    input_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "noisy",
        "noisy_voice1.wav"
    )

    output_folder = os.path.join(
        PROJECT_ROOT,
        "data",
        "enhanced"
    )

    output_file = os.path.join(
        output_folder,
        "spectral_subtraction_voice1.wav"
    )

    # Create enhanced folder
    os.makedirs(
        output_folder,
        exist_ok=True
    )

    print("\n===== TESTING SPECTRAL SUBTRACTION =====")

    # Check input
    if not os.path.exists(input_file):
        print("ERROR: Input audio not found!")
        print(input_file)
        return

    try:

        reduce_noise(
            input_file,
            output_file
        )

        if os.path.exists(output_file):

            print("\n==========================")
            print("✓ TEST PASSED")
            print("==========================")
            print("Output file:")
            print(output_file)

        else:

            print("\n✗ TEST FAILED")
            print("Output file was not created.")

    except Exception as error:

        print("\n✗ ERROR:")
        print(error)


if __name__ == "__main__":
    test_spectral_subtraction()