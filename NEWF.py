from VAD import VAD
from model_inference import REG
import numpy as np
import librosa
import audio_processing as ap
import utils
import soundfile as sf
from glob import iglob
from tqdm import tqdm
import os

def wiener_filter(mag_src, mag_noise):
    max_att = 20 #15
    att_lin = 10 ** (-max_att / 20.0)
    overest = 10 #5

    S_bb = np.mean(np.square(mag_noise), axis=1)
    S_bb = overest * S_bb
    H = np.maximum(att_lin, 1 - np.divide(S_bb, np.square(mag_src)))
    filtered_src = np.multiply(H, mag_src)
    return filtered_src

def set_model():
    vad = VAD("0")
    vad.build(reuse=False)
    vad_sess = vad.init(model_path="./VAD_model/220126/")

    NEWF = REG("0")
    NEWF.build(reuse=False)
    NEWF_sess = NEWF.init(model_path="./NEWF_model/220425/")
    return vad, vad_sess, NEWF, NEWF_sess

def denoise(vad, vad_sess, NEWF, NEWF_sess, audio_file, dst_file):
    y, _ = librosa.load(audio_file, 16000)
    y = ap.second_order_filter_freq(y)
    mel_spec = utils.audio2spec(y, forward_backward=None, SEQUENCE=False, norm=True, hop_length=128,
                     under4k_dim=False, mel_freq=True)
    mel_spec = np.expand_dims(np.expand_dims(mel_spec.T, 0), 3)
    vad_result = vad.predict(vad_sess, mel_spec)

    # spec = utils.audio2spec(y, forward_backward=None, SEQUENCE=False, norm=False, hop_length=128,
    #                  under4k_dim=False, mel_freq=False)
    spec = librosa.stft(y, 256, 128, 256, center=False)
    mag = np.abs(spec)
    phase = spec/mag
    mag = mag.T
    enhance_mag = np.zeros(mag.shape)
    time_step, feature_size = mag.shape
    # spec = np.expand_dims(np.expand_dims(spec.T, 0), 3) # [1, feature_size, time, 1]

    noise_mean = np.zeros(feature_size)
    var = np.zeros(feature_size)
    threshold = 0.8
    alpha = 0.15 #0.15

    obs_frame = 2
    mask_hist = np.zeros([feature_size, obs_frame])

    for i in range(1, time_step-1):
        mask_hist[:, :obs_frame-1] = mask_hist[:, 1:]

        if vad_result[0, i, 0]>threshold:
            input = np.expand_dims(np.expand_dims((mag[i-1:i+2, :] - noise_mean).T, 0), 3)
            mask = NEWF.inference(NEWF_sess, input)[0, 1, :]

            mask_hist[:, obs_frame-1] = mask
            mask_min = np.min(mask_hist, 1)

            enhance_mag[i] = wiener_filter(mag[i], (1-mask_hist)*np.expand_dims(mag[i], 1))
            # enhance_mag[i] = mag[i]*mask_min
            noise = mag[i]*(1-mask_min)
            noise_mean = alpha*noise+(1-alpha)*noise_mean
        else:
            noise_mean = alpha * mag[i] + (1 - alpha) * noise_mean
            input = np.expand_dims(np.expand_dims((mag[i - 1:i + 2, :] - noise_mean).T, 0), 3)
            mask = NEWF.inference(NEWF_sess, input)[0, 1, :]

            mask_hist[:, obs_frame-1] = mask
            mask_min = np.min(mask_hist, 1)

            enhance_mag[i] = wiener_filter(mag[i], (1-mask_hist)*np.expand_dims(mag[i], 1))
            # enhance_mag[i] = mag[i] * mask_min

    enhance_stft = enhance_mag.T*phase
    enhance_y = librosa.istft(enhance_stft, 128, 256)
    # sf.write("/Users/musk/Desktop/test.wav", enhance_y, 16000)
    sf.write(dst_file, enhance_y, 16000)

def batch_denoise(file_path, model_tag):
    audio_file_list = [tag for tag in iglob(file_path)]
    vad, vad_sess, NEWF, NEWF_sess = set_model()

    for audio_file in tqdm(audio_file_list):
        # root_path = "/".join(audio_file.split("/")[:4]) # VADTestSet
        root_path = "/".join(audio_file.split("/")[:5])  # NR0Test
        dst_file_path = "{}/enhance/{}/{}".format(root_path, model_tag, audio_file.split("/")[-2])
        dst_file = "{}/{}".format(dst_file_path, audio_file.split("/")[-1])

        if not os.path.exists(root_path+"/enhance/"):
            os.mkdir(root_path+"/enhance/")
        if not os.path.exists(root_path + "/enhance/{}/".format(model_tag)):
            os.mkdir(root_path + "/enhance/{}/".format(model_tag))
        if not os.path.exists(dst_file_path):
            os.mkdir(dst_file_path)

        denoise(vad, vad_sess, NEWF, NEWF_sess, audio_file, dst_file)

if __name__=="__main__":
    # audio_file = "/Users/musk/dataset/VADTestSet/noisy/0dB/G07FM0210082_h_babble.wav"

    # file_path = "/Users/musk/dataset/VADTestSet/noisy/5dB/*"
    file_path = "/Users/musk/Desktop/NR_Test/NR0/*"
    model_tag = "220425"
    batch_denoise(file_path, model_tag)


