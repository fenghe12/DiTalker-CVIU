import torch
from torch import nn
from torch.nn import functional as F
# import audio
from os.path import dirname, join, basename, isfile
from glob import glob
import os
syncnet_T = 5
syncnet_mel_step_size = 16
class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value

# Default hyperparameters
hparams = HParams(
	num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
	#  network
	rescale=True,  # Whether to rescale audio prior to preprocessing
	rescaling_max=0.9,  # Rescaling value
	
	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	
	n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
	hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
	win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
	
	frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
	
	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,
	# Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
	symmetric_mels=True,
	# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
	# faster and cleaner convergence)
	max_abs_value=4.,
	# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
	# be too big to avoid gradient explosion, 
	# not too small for fast convergence)
	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
	preemphasize=True,  # whether to apply filter
	preemphasis=0.97,  # filter coefficient.
	
	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55,
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,  # To be increased/reduced depending on data.

	###################### Our training parameters #################################
	img_size=96,
	fps=25,
	
	batch_size=16,
	initial_learning_rate=1e-4,
	nepochs=200000000000000000,  ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
	num_workers=16,
	checkpoint_interval=3000,
	eval_interval=3000,
    save_optimizer_state=True,

    syncnet_wt=0.0, # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence. 
	syncnet_batch_size=64,
	syncnet_lr=1e-4,
	syncnet_eval_interval=10000,
	syncnet_checkpoint_interval=10000,

	disc_wt=0.07,
	disc_initial_learning_rate=1e-4,
)


def hparams_debug_string():
	values = hparams.values()
	hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
	return "Hyperparameters:\n" + "\n".join(hp)


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding

if __name__=="__main__":
    print("hello")
    syncnet = SyncNet_color().to("cuda")
    print(syncnet)
    pass
