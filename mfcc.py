import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import math 
from scipy.fftpack import dct


def pre_emphasis(signal, coefficient=0.97):
    '''对信号进行预加重'''
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def audio2frame(signal, frame_length, frame_step, winfunc=lambda x: np.ones((x,))):
    '''分帧'''
    signal_length = len(signal)
    frame_length = int(round(frame_length)) # 四舍五入
    frame_step = int(round(frame_step))
    # 确保我们至少有1帧
    if signal_length <= frame_length:
        frames_num = 1
    else:
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)
    zeros = np.zeros((pad_length - signal_length))
    pad_signal = np.concatenate((signal, zeros))
    indices = np.tile(np.arange(0, frame_length), (frames_num, 1)) + np.tile(np.arange(0, frames_num * frame_step, frame_step),(frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]
    '''
    加窗
    内部实现:
    frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  
    '''
    frames *= np.hamming(frame_length)

    # win = np.tile(winfunc(frame_length), (frames_num, 1))
    # return frames * win
    return frames


def hz2mel(hz):
    '''把频率hz转化为梅尔频率'''
    return 2595 * np.log10(1 + hz / 700.0)

def mel2hz(mel):
    '''把梅尔频率转化为hz'''
    return 700 * (10 ** (mel / 2595.0) - 1)

def get_filter_banks(filters_num=40, NFFT=512, samplerate=16000, low_freq=0, high_freq=None):
    '''计算梅尔三角间距滤波器，该滤波器在第一个频率和第三个频率处为0，在第二个频率处为1'''
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    # 我们要做n个滤波器组，为此需要n+2个点，这意味着在们需要low_mel和high_mel之间线性间隔n个点
    mel_points = np.linspace(low_mel, high_mel, filters_num + 2)
    hz_points = mel2hz(mel_points)
    bin = np.floor((NFFT + 1) * hz_points / samplerate)
    fbank = np.zeros([filters_num, int(np.floor(NFFT / 2 + 1))])
    for j in range(0, filters_num):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank



signal , sample_rate = librosa.load('chew.wav', sr=22050)
emphasized_signal = pre_emphasis(signal)

frames = audio2frame(signal, 0.025*sample_rate, 0.01*sample_rate)
# 傅立叶变换和功率谱

NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum,**2是平方
fbank = get_filter_banks(filters_num=26, samplerate=sample_rate, high_freq=sample_rate / 2)

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

plt.title("filter_banks")
plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1,
		   extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])  # 画热力图
plt.xlabel("Frames", fontsize=14)
plt.ylabel("Dimension", fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.savefig('filter_banks.png')
plt.show()

num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
 
cep_lifter=22
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift  

plt.title("mfcc")
plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.05, extent=[0,mfcc.shape[1],0,mfcc.shape[0]]) #画热力图
plt.xlabel("Frames",fontsize = 14)
plt.ylabel("Dimension",fontsize = 14)
plt.tick_params(axis='both',labelsize = 14)
plt.savefig('mfcc.png')
plt.show()
