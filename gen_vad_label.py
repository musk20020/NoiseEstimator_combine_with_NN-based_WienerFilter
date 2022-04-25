from VAD import VAD
from glob import iglob
import librosa
import audio_processing as ap
import numpy as np
from utils import *
import os
from configuration import get_config

config = get_config()

vad = VAD(gpu_num="0", config=config)

vad.build(reuse=False)
sess = vad.init()

path = "/AudioProject/dataset/TCC300_rename/Dev/*/*"
vad_label_path = "/AudioProject/dataset/TCC300_rename/Dev_vad_label/"
if not os.path.exists(vad_label_path):
    os.makedirs(vad_label_path)

file_list = [tag for tag in iglob(path)]

for file in file_list:
# file = file_list[0]
    y, _ = librosa.load(file, sr=16000, mono=True)
    y = ap.second_order_filter_freq(y)
    y /= np.max(np.abs(y))

    mag_norm = audio2spec(y, forward_backward=False, SEQUENCE=False, norm=True,
                          hop_length=128, under4k_dim=False, mel_freq=True)
    mag_norm = np.expand_dims(np.expand_dims(mag_norm, 0), 3)
    predict = vad.predict(sess, mag_norm)
    speaker = file.split("/")[-2]
    audio = file.split("/")[-1].split(".")[0]
    # print(file)
    # print(speaker)
    # print(audio)
    # print(predict[0, :, 0])
    speaker_label_path = "{}{}".format(vad_label_path, speaker)
    if not os.path.exists(speaker_label_path):
        os.makedirs(speaker_label_path)
    label_dst = "{}/{}.npy".format(speaker_label_path, audio)
    np.save(label_dst, predict[0, :, 0])