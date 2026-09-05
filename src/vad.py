import numpy as np


def voice_activity_detection(
    audio,
    sample_rate,
    frame_duration=0.02,
    threshold=0.01
):

    frame_size = int(sample_rate * frame_duration)

    voice_activity = []

    for start in range(
        0,
        len(audio),
        frame_size
    ):

        frame = audio[start:start + frame_size]

        if len(frame) == 0:
            continue

        energy = np.mean(frame ** 2)

        if energy > threshold:
            voice_activity.append(True)

        else:
            voice_activity.append(False)

    return voice_activity