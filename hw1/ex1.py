
import sounddevice as sd
from scipy.io.wavfile import write
from time import time
import os 
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf 


class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram


class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram




class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        dbFSthres, 
        duration_thres
    ):
        self.frame_length_in_s = frame_length_in_s
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_length_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.dbFSthres = dbFSthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        log_mel_spec = self.mel_spec_processor.get_mel_spec(audio)
        dbFS = 20 * log_mel_spec
        energy = tf.math.reduce_mean(dbFS, axis=1)

        non_silence = energy > self.dbFSthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = (non_silence_frames + 1) * self.frame_length_in_s

        if non_silence_duration > self.duration_thres:
            return 0
        else:
            return 1



parser = argparse.ArgumentParser()
parser.add_argument('--device',type=int, default=0)

args = parser.parse_args()

deviceID = args.device

samplerate = 16000
n_channels = 1
resolution = 'int16'

# Samples in 0.5 sec
blocksize_check = 8000 

# audio buffer contains 1 sec of audio 
total_blocksize = 16000 



def callback(indata, frames, callback_time, status):

    global audio_buffer
    global vad_processor
    
    timestamp = time()
    
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.int16)

    audio_buffer = tf.concat([audio_buffer[blocksize_check:], tf_indata[:,0]],axis=0)

    audio_float32 = tf.cast(audio_buffer,tf.float32)
    audio_normalized = audio_float32 / tf.int16.max
   
    if not vad_processor.is_silence(audio_normalized):
        write(f"{timestamp}.wav",samplerate,audio_normalized.numpy())

    
# Instantiate VAD class with parameters of ex. 1.1
vad_processor = VAD(16000, 0.05, 80, 0, 8000, -35, 0.2)

audio_buffer = tf.zeros(samplerate,dtype='int16')


with sd.InputStream(device=0, samplerate=16000, dtype='int16', channels=1, callback=callback, blocksize=8000):
    
    print("It's recording ...")
    while True:
        key = input()
        if key in ('q','Q'):
            print('Stop recording')
            break
            

