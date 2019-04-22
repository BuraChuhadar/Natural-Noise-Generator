import os
import numpy as np
from librosa.output import write_wav as wav_w
from librosa.core import load as wav_r


def main():
    file_path = "beach/beach.wav"
    output_dir = "beach_sliced"
    sample_rate = 11025
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for i in range(20):
        demo_wave = wav_r(file_path,sr=sample_rate,offset=i,duration=32768/sample_rate)[0]
        wav_w(os.path.join(output_dir,str(i)+'.wav'), demo_wave, sample_rate)





if __name__ == '__main__':
    main()
