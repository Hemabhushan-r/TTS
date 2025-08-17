import numpy as np
import os
from typing import Optional, Tuple, Union,List
from dataclasses import asdict, dataclass


from coqpit import Coqpit, check_argument


os.environ['LIGHTNING_USER_ID']='bcd4694e-df09-4161-a4de-1d3f1491d6e3'
os.environ['LIGHTNING_API_KEY']='06100d14-9a2d-4723-98ff-57fe25eebca5'
# os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'


import pandas as pd
from datasets import Dataset,load_dataset
from collections import Counter
from lightning_sdk import Studio
import torch
import torchaudio
from sklearn.preprocessing import LabelEncoder

from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import functools
from coqpit import Coqpit
import time

from torch.utils.tensorboard import SummaryWriter
import gc
from huggingface_hub import login,hf_hub_download
from huggingface_hub import HfApi, HfFolder
import shutil
from prettytable import PrettyTable
import psutil
import time
import threading
import sys
import traceback
import subprocess
import shutil


from tqdm import tqdm

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from lightning.pytorch.callbacks import ModelSummary

from lightning.pytorch.profilers import PyTorchProfiler

import torch.utils.checkpoint as checkpoint

from dotenv import load_dotenv 


DATALOADER_NUM_PROC=4
BATCH_SIZE=32
MAX_AUDIO_DURATION=15
DTYPE_PT=torch.float32


@dataclass
class BaseAudioConfig(Coqpit):
    '''Base config to definge audio processing parameters. It is used to initialize
    ```TTS.utils.audio.AudioProcessor.```

    Args:
        fft_size (int):
            Number of STFT frequency levels aka.size of the linear spectogram frame. Defaults to 1024.

        win_length (int):
            Each frame of audio is windowed by window of length ```win_length``` and then padded with zeros to match
            ```fft_size```. Defaults to 1024.

        hop_length (int):
            Number of audio samples between adjacent STFT columns. Defaults to 1024.

        frame_shift_ms (int):
            Set ```hop_length``` based on milliseconds and sampling rate.

        frame_length_ms (int):
            Set ```win_length``` based on milliseconds and sampling rate.

        stft_pad_mode (str):
            Padding method used in STFT. 'reflect' or 'center'. Defaults to 'reflect'.

        sample_rate (int):
            Audio sampling rate. Defaults to 22050.

        resample (bool):
            Enable / Disable resampling audio to ```sample_rate```. Defaults to ```False```.

        preemphasis (float):
            Preemphasis coefficient. Defaults to 0.0.

        ref_level_db (int): 20
            Reference Db level to rebase the audio signal and ignore the level below. 20Db is assumed the sound of air.
            Defaults to 20.

        do_sound_norm (bool):
            Enable / Disable sound normalization to reconcile the volume differences among samples. Defaults to False.

        log_func (str):
            Numpy log function used for amplitude to DB conversion. Defaults to 'np.log10'.

        do_trim_silence (bool):
            Enable / Disable trimming silences at the beginning and the end of the audio clip. Defaults to ```True```.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        pitch_fmax (float, optional):
            Maximum frequency of the F0 frames. Defaults to ```640```.

        pitch_fmin (float, optional):
            Minimum frequency of the F0 frames. Defaults to ```1```.

        trim_db (int):
            Silence threshold used for silence trimming. Defaults to 45.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        power (float):
            Exponent used for expanding spectrogra levels before running Griffin Lim. It helps to reduce the
            artifacts in the synthesized voice. Defaults to 1.5.

        griffin_lim_iters (int):
            Number of Griffing Lim iterations. Defaults to 60.

        num_mels (int):
            Number of mel-basis frames that defines the frame lengths of each mel-spectrogram frame. Defaults to 80.

        mel_fmin (float): Min frequency level used for the mel-basis filters. ~50 for male and ~95 for female voices.
            It needs to be adjusted for a dataset. Defaults to 0.

        mel_fmax (float):
            Max frequency level used for the mel-basis filters. It needs to be adjusted for a dataset.

        spec_gain (int):
            Gain applied when converting amplitude to DB. Defaults to 20.

        signal_norm (bool):
            enable/disable signal normalization. Defaults to True.

        min_level_db (int):
            minimum db threshold for the computed melspectrograms. Defaults to -100.

        symmetric_norm (bool):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else
            [0, k], Defaults to True.

        max_norm (float):
            ```k``` defining the normalization range. Defaults to 4.0.

        clip_norm (bool):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        stats_path (str):
            Path to the computed stats file. Defaults to None.
    '''

    # stft parameters
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    # audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    # silence trimming
    do_trim_silence: bool = True
    trim_db: int = 45
    # rms volume normalization
    do_rms_norm: bool = False
    db_level: float = None
    # griffin-lim params
    power: float = 1.5
    griffin_lim_iters: int = 60
    # mel-spec params
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # f0 params
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        '''Check config fields'''
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)




class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)

class BaseEncoder(nn.Module):
    '''Base `encoder` class. Every new `encoder` model must inherit this.

    It defines common `encoder` specific functions.
    '''

    # pylint: disable=W0102
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            # TorchSTFT(
            #     n_fft=audio_config["fft_size"],
            #     hop_length=audio_config["hop_length"],
            #     win_length=audio_config["win_length"],
            #     sample_rate=audio_config["sample_rate"],
            #     window="hamming_window",
            #     mel_fmin=0.0,
            #     mel_fmax=None,
            #     use_htk=True,
            #     do_amp_to_db=False,
            #     n_mels=audio_config["num_mels"],
            #     power=2.0,
            #     use_mel=True,
            #     mel_norm=None,
            # )
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    @torch.no_grad()
    def inference(self, x, l2_norm=True):
        return self.forward(x, l2_norm)

    @torch.no_grad()
    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True):
        '''
        Generate embeddings for a batch of utterances
        x: 1xTxD
        '''
        # map to the waveform size
        if self.use_torch_spec:
            num_frames = num_frames * self.audio_config["hop_length"]

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings

    def get_criterion(self, c: Coqpit, num_classes=None):
        if c.loss == "ge2e":
            criterion = GE2ELoss(loss_method="softmax")
        elif c.loss == "angleproto":
            criterion = AngleProtoLoss()
        elif c.loss == "softmaxproto":
            criterion = SoftmaxAngleProtoLoss(c.model_params["proj_dim"], num_classes)
        else:
            raise Exception("The %s  not is a loss supported" % c.loss)
        return criterion

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str,
        eval: bool = False,
        use_cuda: bool = False,
        criterion=None,
        cache=False,
    ):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache) #Needs func from coqui Repo
        try:
            self.load_state_dict(state["model"])
            print(" > Model fully restored. ")
        except (KeyError, RuntimeError) as error:
            # If eval raise the error
            if eval:
                raise error

            print(" > Partial model initialization.")
            model_dict = self.state_dict()
            model_dict = set_init_dict(model_dict, state["model"], c)
            self.load_state_dict(model_dict)
            del model_dict

        # load the criterion for restore_path
        if criterion is not None and "criterion" in state:
            try:
                criterion.load_state_dict(state["criterion"])
            except (KeyError, RuntimeError) as error:
                print(" > Criterion load ignored because of:", error)

        # instance and load the criterion for the encoder classifier in inference time
        if (
            eval
            and criterion is None
            and "criterion" in state
            and getattr(config, "map_classid_to_classname", None) is not None
        ):
            criterion = self.get_criterion(config, len(config.map_classid_to_classname))
            criterion.load_state_dict(state["criterion"])

        if use_cuda:
            self.cuda()
            if criterion is not None:
                criterion = criterion.cuda()

        if eval:
            self.eval()
            assert not self.training

        if not eval:
            return criterion, state["step"]
        return criterion
        
     
class LSTMWithProjection(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, proj_size, bias=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        o, (_, _) = self.lstm(x)
        return self.linear(o)


class LSTMWithoutProjection(nn.Module):
    def __init__(self, input_dim, lstm_dim, proj_dim, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim, num_layers=num_lstm_layers, batch_first=True)
        self.linear = nn.Linear(lstm_dim, proj_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.relu(self.linear(hidden[-1]))


class LSTMSpeakerEncoder(BaseEncoder):
    def __init__(
        self,
        input_dim,
        proj_dim=256,
        lstm_dim=768,
        num_lstm_layers=3,
        use_lstm_with_projection=True,
        use_torch_spec=False,
        audio_config=None,
    ):
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        layers = []
        # choise LSTM layer
        if use_lstm_with_projection:
            layers.append(LSTMWithProjection(input_dim, lstm_dim, proj_dim))
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection(proj_dim, lstm_dim, proj_dim))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = LSTMWithoutProjection(input_dim, lstm_dim, proj_dim, num_lstm_layers)

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = self.get_torch_mel_spectrogram_class(audio_config)
        else:
            self.torch_spec = None

        self._init_layers()

    def _init_layers(self):
        for name, param in self.layers.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, l2_norm=True):
        '''Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        '''
        # intermed=''
        # intermed+=f'Input {x} '
        
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     raise ValueError("Input contains NaN or Inf values")
        
        with torch.no_grad():
            # with torch.cuda.amp.autocast(enabled=False):
            if self.use_torch_spec:
                x.squeeze_(1)

                # with open('/kaggle/working/lstm_enc_test.txt','w') as f:
                #     f.write(f'x before torch spec {x.dtype}')
                x = self.torch_spec(x)

                # with open('/kaggle/working/lstm_enc_test.txt','a') as f:
                #     f.write(f'x after torch spec {x.dtype}')
                # if torch.isnan(x).any() or torch.isinf(x).any():
                #     raise ValueError("Input after mel contains NaN or Inf values")
            # intermed+=f'after torch spec {x} '
            x = self.instancenorm(x).transpose(1, 2)

            # with open('/kaggle/working/lstm_enc_test.txt','a') as f:
            #     f.write(f'x after instance norm {x.dtype}')
            # if torch.isnan(x).any() or torch.isinf(x).any():
            #     raise ValueError("Input after norm contains NaN or Inf values")
        # intermed+=f'After norm {x} '
        d = self.layers(x)

        # with open('/kaggle/working/lstm_enc_test.txt','a') as f:
        #     f.write(f'd after layers {d.dtype}')
        # if torch.isnan(d).any() or torch.isinf(d).any():
        #     raise ValueError("Input after layers contains NaN or Inf values")
        # intermed+=f'after layers {d} '
        if self.use_lstm_with_projection:
            d = d[:, -1]
        # intermed+=f'after projection {d} '
        if l2_norm:
            d = torch.nn.functional.normalize(d, p=2, dim=1)
        # intermed+=f'after l2 norm {d} '
        
        # with open('/kaggle/working/intermed.txt','w') as f:
        #     f.write(intermed)
        return d


def model_summary(models_dict):
    table = PrettyTable(["Modules","Class","Parameters"])
    models_keys = list(models_dict.keys())
    models_list = list(models_dict.values())
    for i,model in enumerate(models_list):
        total_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            total_params += params
        table.add_row([models_keys[i],type(models_list[i]).__name__, total_params])
    print(table)


class AngleProtoLoss(nn.Module):
    '''
    Implementation of the Angular Prototypical loss defined in https://arxiv.org/abs/2003.11982
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector
        Args:
            - init_w (float): defines the initial value of w
            - init_b (float): definies the initial value of b
    '''

    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        # pylint: disable=E1102
        self.w = nn.Parameter(torch.tensor(init_w))
        # pylint: disable=E1102
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print(" > Initialized Angular Prototypical loss")

    def forward(self, x, _label=None):
        '''
        Calculates the AngleProto loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''

        assert x.size()[1] >= 2

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        num_speakers = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(
            out_positive.unsqueeze(-1).expand(-1, -1, num_speakers),
            out_anchor.unsqueeze(-1).expand(-1, -1, num_speakers).transpose(0, 2),
        )
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        label = torch.arange(num_speakers).to(cos_sim_matrix.device)
        L = self.criterion(cos_sim_matrix, label)
        return L


class SoftmaxLoss(nn.Module):
    '''
    Implementation of the Softmax loss as defined in https://arxiv.org/abs/2003.11982
        Args:
            - embedding_dim (float): speaker embedding dim
            - n_speakers (float): number of speakers
    '''

    def __init__(self, embedding_dim, n_speakers):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(embedding_dim, n_speakers)

        print("Initialised Softmax Loss")

    def forward(self, x, label=None):
        # reshape for compatibility
        x = x.reshape(-1, x.size()[-1])
        label = label.reshape(-1)

        x = self.fc(x)
        L = self.criterion(x, label)

        return L

    def inference(self, embedding):
        x = self.fc(embedding)
        activations = torch.nn.functional.softmax(x, dim=1).squeeze(0)
        class_id = torch.argmax(activations)
        return class_id


class SoftmaxAngleProtoLoss(nn.Module):
    '''
    Implementation of the Softmax AnglePrototypical loss as defined in https://arxiv.org/abs/2009.14153
        Args:
            - embedding_dim (float): speaker embedding dim
            - n_speakers (float): number of speakers
            - init_w (float): defines the initial value of w
            - init_b (float): definies the initial value of b
    '''

    def __init__(self, embedding_dim, n_speakers, init_w=10.0, init_b=-5.0):
        super().__init__()

        self.softmax = SoftmaxLoss(embedding_dim, n_speakers)
        self.angleproto = AngleProtoLoss(init_w, init_b)

        print("Initialised SoftmaxAnglePrototypical Loss")

    def forward(self, x, label=None):
        '''
        Calculates the SoftmaxAnglePrototypical loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''

        Lp = self.angleproto(x)

        Ls = self.softmax(x, label)

        return Ls + Lp



class GE2ELoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        '''
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]

        Accepts an input of size (N, M, D)

            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)

        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        '''
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        ''' 
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''
        #Calculate centroids
        centroids = torch.mean(dvecs, 1)

        #Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()

        



def download_file(example):
    filepath = example['File Path']
    filepath_split = filepath.split("/")
    filepath_join = "/".join(filepath_split[5:])
    filepath_join = "/"+filepath_join
    filedir ="/"+"/".join(filepath_split[6:-1])
    os.makedirs(filedir, exist_ok=True)
    temp_dir=f'/teamspace/studios/this_studio/temp_dir'
    target_path = f'/kaggle/tmp/files{filepath_join}'
    temp_source_dir=f"{temp_dir}/{filepath_split[-1]}"
    operation_1=f"cp '{filepath}' '{temp_source_dir}'"
    operation_2=f"rm -f '{temp_source_dir}'"
    #print(filepath)
    #print(target_path)
    
    remote_control.run(operation_1)
    remote_control.download_file(temp_source_dir,target_path)
    remote_control.run(operation_2)
    return target_path

# Function to load a .wav file and process it
def load_audio(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    print(f'Sample rate:{sample_rate}')
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform.squeeze().numpy(), 16000


def load_partial_audio(wav_file,input_audio_sr, duration_sec=5, start_sec=0, target_sr=16000):
    # Get audio file info to check the number of frames
    info = torchaudio.info(wav_file)
    
    # Total frames in the audio file
    total_frames = info.num_frames
    
    # Calculate the number of frames to load
    frame_offset = start_sec * input_audio_sr
    num_frames = min(duration_sec * input_audio_sr, total_frames - frame_offset)
    
    # Calculate the number of frames to load based on the duration and sample rate
    waveform, sample_rate = torchaudio.load(wav_file, frame_offset=frame_offset, num_frames=num_frames)
    
    # Resample if the audio is not in the target sample rate
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)
    return waveform.squeeze().numpy(), target_sr

def load_partial_audio_from_wav(waveform, input_audio_sr, duration_sec=5, start_sec=0, target_sr=16000):
    # Ensure waveform is a numpy array for processing
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Total frames in the audio data
    total_frames = waveform.shape[-1]

    # Calculate frame offsets
    frame_offset = int(start_sec * input_audio_sr)
    num_frames = min(int(duration_sec * input_audio_sr), total_frames - frame_offset)
    
    # Extract the portion of the waveform
    waveform_slice = waveform[:, frame_offset:frame_offset + num_frames]

    # Resample if necessary
    if input_audio_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=input_audio_sr, new_freq=target_sr)
        waveform_tensor = torch.tensor(waveform_slice,dtype=DTYPE_PT)
        waveform_resampled = resampler(waveform_tensor)
        waveform_slice = waveform_resampled.numpy()

    return waveform_slice.squeeze(), target_sr




class SpeakerSamplerPrev(data.Sampler):
    def __init__(self, data_source, num_speakers=4, utterances_per_speaker=2,split=None, num_replicas=None, rank=None):
        self.data_source = data_source
        self.num_speakers = num_speakers
        self.utterances_per_speaker = utterances_per_speaker
        self.speakers_grouped = self.data_source.groupby('SpeakerID')
        self.unique_speakers = list(self.speakers_grouped.groups.keys())
        self.cust_len = len(self.data_source)
        self.split=split
        
        # Delay distributed-related initialization until DDP is launched
        self.num_replicas = None
        self.rank = None

        
    def _setup_ddp(self):
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1  # In case not running distributed
            self.rank = 0

        # Split speakers between processes based on rank
        self.process_speakers = self.unique_speakers[self.rank::self.num_replicas]
        
    def check_speaker(self,speaker_dict):
        if len(speaker_dict.keys()) < self.num_speakers:
            return False

        keys_to_be_deleted=[]

        for speaker in speaker_dict.keys():
            if len(speaker_dict[speaker])<self.utterances_per_speaker:
                keys_to_be_deleted.append(speaker)

        for speaker in keys_to_be_deleted:
            del speaker_dict[speaker]
            
        if len(speaker_dict.keys()) < self.num_speakers:
            return False
        
        return True


    def __iter__(self):
        if self.num_replicas is None:
            self._setup_ddp()
        speaker_dict = dict()
        for speaker in self.process_speakers:  # Only work with speakers for this process
            speaker_dict[speaker] = self.speakers_grouped.get_group(speaker).index.tolist()

        indices = []

        while self.check_speaker(speaker_dict):
            selected_speakers = np.random.choice(list(speaker_dict.keys()), self.num_speakers, replace=False)
            for speaker in selected_speakers:
                speaker_indices = speaker_dict[speaker]
                indices.extend(speaker_indices[:self.utterances_per_speaker])
                speaker_dict[speaker] = speaker_indices[self.utterances_per_speaker:]
        
        
        if self.split=="train":
            end_index=len(indices)-self.utterances_per_speaker*self.num_speakers
            indices = indices[:end_index]
        else:
            start_index=-self.utterances_per_speaker*self.num_speakers
            indices = indices[start_index:]
        self.cust_len = len(indices)
        print('Sampler Len',self.split,len(indices))
        return iter(indices)

    def __len__(self):
        # Length now reflects this process's speaker subset
        return self.cust_len // (self.num_replicas or 1)


class SpeakerSampler(data.Sampler):
    def __init__(self, data_source, num_speakers=4, utterances_per_speaker=2,split=None, num_replicas=None, rank=None):
        self.data_source = data_source
        self.num_speakers = num_speakers
        self.utterances_per_speaker = utterances_per_speaker
        self.speakers_grouped = self.data_source.groupby('SpeakerID')
        self.unique_speakers = list(self.speakers_grouped.groups.keys())
        self.cust_len = len(self.data_source)
        self.split=split
        
        # Delay distributed-related initialization until DDP is launched
        self.num_replicas = None
        self.rank = None

        
    def _setup_ddp(self):
        if dist.is_available() and dist.is_initialized() and self.split=="train": # Very imp to set split
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1  # In case not running distributed
            self.rank = 0

        # Split speakers between processes based on rank
        self.process_speakers = self.unique_speakers[self.rank::self.num_replicas]
        
    def check_speaker(self,speaker_dict):
        if len(speaker_dict.keys()) < self.num_speakers:
            return False

        keys_to_be_deleted=[]

        for speaker in speaker_dict.keys():
            if len(speaker_dict[speaker])<self.utterances_per_speaker:
                keys_to_be_deleted.append(speaker)

        for speaker in keys_to_be_deleted:
            del speaker_dict[speaker]
            
        if len(speaker_dict.keys()) < self.num_speakers:
            return False
        
        return True


    def __iter__(self):
        if self.num_replicas is None:
            self._setup_ddp()
        speaker_dict = dict()
        for speaker in self.unique_speakers:  # Work with all speakers and then split
            speaker_dict[speaker] = self.speakers_grouped.get_group(speaker).index.tolist()
        
        print(f'speakers {len(speaker_dict.keys())}')
        print(f'unique speakers {len(self.unique_speakers)}')
        indices = []

        while self.check_speaker(speaker_dict):
            selected_speakers = np.random.choice(list(speaker_dict.keys()), self.num_speakers, replace=False)
            for speaker in selected_speakers:
                speaker_indices = speaker_dict[speaker]
                indices.extend(speaker_indices[:self.utterances_per_speaker])
                speaker_dict[speaker] = speaker_indices[self.utterances_per_speaker:]
        
        # print(f'indices {indices}')
        
        if self.split=="train":
            end_index=len(indices)-(self.utterances_per_speaker*self.num_speakers)
            indices = indices[:end_index]
        else:
            start_index=(-self.utterances_per_speaker*self.num_speakers)
            indices = indices[start_index:]
            
        print(f'Before split Mod Check {len(indices)}',(len(indices)/(self.utterances_per_speaker*self.num_speakers)))
        if (len(indices)%(self.utterances_per_speaker*self.num_speakers*self.num_replicas))==0:
            per_device_split_size= len(indices)//self.num_replicas
            start_index = self.rank*per_device_split_size
            end_index = start_index+per_device_split_size
            indices = indices[start_index:end_index]
        else:
            while (len(indices)%(self.utterances_per_speaker*self.num_speakers*self.num_replicas))!=0:
                index=-self.utterances_per_speaker*self.num_speakers
                indices = indices+indices[index:]
            per_device_split_size= len(indices)//self.num_replicas
            start_index = self.rank*per_device_split_size
            end_index = start_index+per_device_split_size
            indices = indices[start_index:end_index]
        self.cust_len = len(indices)
        print('Sampler Len',self.split,len(indices))
        return iter(indices)

    def __len__(self):
        # Length now reflects this process's speaker subset
        return self.cust_len // (self.num_replicas or 1)







class KannadaAudioDataset(data.Dataset):
    def __init__(self,ds,source_sample_rate=48000):
        self.ds=ds
        self.source_sample_rate=source_sample_rate
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self,idx):
        sample = self.ds[idx]
        sample_path='/kaggle/working/SampleKnDataset/LDC-IL_Scheduled_Kannada_Female_21To50_Creative Text-T2_SP-0131_T2-0006.wav'
        waveform_np,sr=load_partial_audio(sample_path,self.source_sample_rate, duration_sec=MAX_AUDIO_DURATION)
        # speaker_labels=torch.tensor(-1)
        speaker_labels = torch.tensor(sample['SpeakerLabel'])
        sample_obtained=-1
        try:
            raise ValueError("Just Error")
            sample_path = download_file(sample)
            waveform_np,sr = load_partial_audio(sample_path,self.source_sample_rate, duration_sec=MAX_AUDIO_DURATION)
            # waveform_tensor = torch.tensor(waveform_np)
            sample_obtained=1
            os.remove(sample_path)
            sop=f'obtained sample {idx}'
            # with open('/kaggle/working/sample_check.txt','w') as f:
            #     f.write(sop)
            return sample_path,torch.tensor(waveform_np,dtype=DTYPE_PT).mean(dim=0, keepdim=True),speaker_labels,sample_obtained
        except Exception as e:
            sample_obtained=1
            file_op=f'Sample {sample} failed with exception {e} '
            file_op+=f'{traceback.format_exc()}'
            with open('/kaggle/working/exception.txt','w') as f:
                f.write(file_op)
            return sample_path,torch.tensor(waveform_np,dtype=DTYPE_PT).mean(dim=0, keepdim=True),speaker_labels,sample_obtained


class KannadaAudioDatasetHF(data.Dataset):
    def __init__(self,metadata_ds,ds,source_sample_rate=48000):
        self.metadata_ds=metadata_ds
        self.ds=ds
        self.source_sample_rate=source_sample_rate
        self.dummy_sample_path='/kaggle/working/SampleKnDataset/LDC-IL_Scheduled_Kannada_Female_21To50_Creative Text-T2_SP-0131_T2-0006.wav'
        waveform_np,sr=load_partial_audio(self.dummy_sample_path,self.source_sample_rate, duration_sec=MAX_AUDIO_DURATION)
        
        self.dumm_waveform_np=waveform_np
        self.sample_rate=sr
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self,idx):
        sample = self.ds[idx]
        sample_metadata = self.metadata_ds[idx]
        sample_path=self.dummy_sample_path
        waveform_np,sr=self.dumm_waveform_np,self.sample_rate
        # speaker_labels=torch.tensor(-1)
        speaker_labels = torch.tensor(sample_metadata['SpeakerLabel'])
        sample_obtained=-1
        try:
            waveform_np,sr = load_partial_audio_from_wav(sample['audio'],self.source_sample_rate,duration_sec=MAX_AUDIO_DURATION)
            # waveform_tensor = torch.tensor(waveform_np)
            sample_obtained=1
            return sample_path,torch.tensor(waveform_np,dtype=DTYPE_PT).mean(dim=0, keepdim=True),speaker_labels,sample_obtained
        except Exception as e:
            sample_obtained=-1
            file_op=f'Sample {sample_metadata} failed with exception {e} '
            file_op+=f'{traceback.format_exc()}'
            with open('/kaggle/working/exception.txt','w') as f:
                f.write(file_op)
            return sample_path,torch.tensor(waveform_np,dtype=DTYPE_PT).mean(dim=0, keepdim=True),speaker_labels,sample_obtained


class KannadaAudioIterableDataset(data.IterableDataset):
    def __init__(self,ds_id,ds_subfolders,source_sample_rate=48000):
        self.ds=ds
        self.source_sample_rate=source_sample_rate
        self.ds_id=ds_id
        self.ds_subfolders=ds_subfolders
        
    def __len__(self):
        return len(self.ds)

    def _get_iter(self):
        for subfolder in self.ds_subfolders:
            curr_ds = load_dataset(self.ds_id,subfolder,streaming=True)

            curr_ds = curr_ds['train']

            sample_path='/kaggle/working/SampleKnDataset/LDC-IL_Scheduled_Kannada_Female_21To50_Creative Text-T2_SP-0131_T2-0006.wav'
            waveform_np_fail,sr=load_partial_audio(sample_path,self.source_sample_rate, duration_sec=MAX_AUDIO_DURATION)
            

            for sample in curr_ds:
                speaker_labels = torch.tensor(sample['SpeakerLabel'])
                try:
                    waveform_np,sr = load_partial_audio_from_wav(np.array(sample['audio']),self.source_sample_rate,duration_sec=MAX_AUDIO_DURATION)
                    sample_obtained=1
                    yield sample_path,torch.tensor(waveform_np,dtype=DTYPE_PT).mean(dim=0, keepdim=True),speaker_labels,sample_obtained
                except Exception as e:
                    sample_obtained=-1
                    yield sample_path,torch.tensor(waveform_np_fail,dtype=DTYPE_PT).mean(dim=0, keepdim=True),speaker_labels,sample_obtained
        
        
    def __iter__(self):
        return self._get_iter()

def ds_collate_function(batch):
    sample_path,waveform_np_arr,speaker_labels,sample_obtained = zip(*batch)
    
    indices = [i for i, num in enumerate(sample_obtained) if num == -1]
    
    speaker_labels_to_be_removed = [speaker_labels[index].item() for index in indices]
    
    speaker_labels_to_be_removed  = list(set(speaker_labels_to_be_removed))
    
    indices_reqd = [i for i,tensor in enumerate(speaker_labels) if tensor.item() not in speaker_labels_to_be_removed]
    
    ds_collate=f'before speaker_labels {speaker_labels}  sample_obtained {sample_obtained}  indices_reqd {indices_reqd} '
    
    sample_path = [sample_path[index] for index in indices_reqd]
    
    waveform_np_arr = [waveform_np_arr[index] for index in indices_reqd]
    
    speaker_labels = [speaker_labels[index] for index in indices_reqd]
    
    ds_collate+=f'after speaker_labels {speaker_labels}  sample_obtained {sample_obtained}  indices_reqd {indices_reqd} '
    
    
    with open('/kaggle/working/ds_collate.txt','a') as f:
        f.write(ds_collate)
    
    
    # Find the maximum number of samples across all tensors
    max_num_samples = max(tensor.shape[1] for tensor in waveform_np_arr)

    # Pad each tensor along the num_samples dimension to match max_num_samples
    padded_tensors = [torch.nn.functional.pad(tensor, (0, max_num_samples - tensor.shape[1])) for tensor in waveform_np_arr]

    # Stack the padded tensors along a new dimension (e.g., batch dimension)
    waveform_list = torch.stack(padded_tensors, dim=0)
    
    # waveform_list = torch.stack(waveform_np_arr)
    
    speaker_labels = torch.stack(speaker_labels)
    
    return sample_path,waveform_list,speaker_labels


class SpeakerEncodingModel(L.LightningModule):
    def __init__(self,
            num_classes=600,
            lr=1e-5,
            weight_decay=0,
            tpu_mesh=None,
            tpu_device=None,
            num_lstm_layers=3,
            num_speaker_per_batch=4,
            num_utterances_per_speaker=2,
            audio_config=BaseAudioConfig(),
            loss_name="angleproto"):
        super().__init__()
        self.num_classes=num_classes
        self.num_lstm_layers=num_lstm_layers
        self.num_speaker_per_batch=num_speaker_per_batch
        self.num_utterances_per_speaker=num_utterances_per_speaker
        self.lr=lr
        self.weight_decay=weight_decay
        self.tpu_mesh=tpu_mesh
        self.tpu_device=tpu_device
        self.lstm_speaker_encoder = None
        self.loss_fn = None
        self.loss_name = loss_name
        self.audio_config = audio_config
        self.writer = SummaryWriter('/teamspace/studios/this_studio/TTS/tensorboard')
    
    def configure_model(self):
        if self.lstm_speaker_encoder is None:
            self.lstm_speaker_encoder=LSTMSpeakerEncoder(input_dim=80,num_lstm_layers=self.num_lstm_layers,use_torch_spec=True,audio_config=self.audio_config,use_lstm_with_projection=True)
        if self.loss_fn is None:
            if self.loss_name == "ge2eloss":
                self.loss_fn = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax')
            elif self.loss_name == "angleproto":
                self.loss_fn = AngleProtoLoss()
        
    def training_step(self,batch,batch_idx):
        sample_path,waveform_np,speaker_labels = batch
        
        
        # embeddings = [self.lstm_speaker_encoder(waveform.to(self.device)) for waveform in waveform_np]
        
        curr_batch_size = waveform_np.shape[0] 
        
        waveform_np = waveform_np.flatten(0,1)
        
        embeddings = self.lstm_speaker_encoder(waveform_np)
        
        # embeddings = [tensor.mean(dim=0) for tensor in embeddings]
        
        # print('Before Stack',embeddings[0].shape)
        
        # print(torch.nn.functional.cosine_similarity(embeddings[0][0].unsqueeze(0),embeddings[0][1].unsqueeze(0)))
        
        # embeddings = torch.stack(embeddings)
        
        embeddings = embeddings.view(curr_batch_size,int(embeddings.shape[0]/curr_batch_size),embeddings.shape[-1])
        
        # print(embeddings.shape)
        
        embeddings = embeddings.view(int(curr_batch_size/self.num_utterances_per_speaker),self.num_utterances_per_speaker,embeddings.shape[-1])
        
        # embeddings_str=f'{batch_idx} {embeddings}'
        
        # with open('/kaggle/working/train_embed.txt','w') as f:
        #     f.write(embeddings_str)
        
        loss = self.loss_fn(embeddings)

        self.log('Training Loss', loss.detach().item(), on_step=True, on_epoch=True, logger=True)
        
        # self.writer.add_scalar("Training Loss",loss.detach().item(),self.trainer.global_step)
        
        # loss_val =f'{batch_idx} {loss.detach().item()}'
        
        # with open('/kaggle/working/train_loss.txt','w') as f:
        #     f.write(loss_val)
        
        return loss
    
    def validation_step(self,batch,batch_idx):
        sample_path,waveform_np,speaker_labels = batch
        
        
        # embeddings = [self.lstm_speaker_encoder(waveform.to(self.device)) for waveform in waveform_np]
        
        curr_batch_size = waveform_np.shape[0] 
        
        waveform_np = waveform_np.flatten(0,1)
        
        embeddings = self.lstm_speaker_encoder(waveform_np)
        
        # embeddings = [tensor.mean(dim=0) for tensor in embeddings]
        
        # print('Before Stack',embeddings[0].shape)
        
        # print(torch.nn.functional.cosine_similarity(embeddings[0][0].unsqueeze(0),embeddings[0][1].unsqueeze(0)))
        
        # embeddings = torch.stack(embeddings)
        
        embeddings = embeddings.view(curr_batch_size,int(embeddings.shape[0]/curr_batch_size),embeddings.shape[-1])
        
        # print(embeddings.shape)
        
        embeddings = embeddings.view(int(curr_batch_size/self.num_utterances_per_speaker),self.num_utterances_per_speaker,embeddings.shape[-1])
        
        # embeddings_str=f'{batch_idx} {embeddings}'
        
        # with open('/kaggle/working/val_embed.txt','w') as f:
        #     f.write(embeddings_str)
        
        loss = self.loss_fn(embeddings)

        self.log('Validation Loss', loss.detach().item(), on_step=True, on_epoch=True, logger=True)

        # self.writer.add_scalar("Validation Loss",loss.detach().item(),self.trainer.global_step)
        
        # loss_val =f'{batch_idx} {loss.detach().item()}'
        
        # with open('/kaggle/working/val_loss.txt','w') as f:
        #     f.write(loss_val)
        
        return loss
    
    def configure_optimizers(self):
        # params= list(self.wav2vec2model.parameters())+list(self.linear_layer.parameters())
        # params= list(self.wav2vec2model.parameters())
        params  = [{'params':self.lstm_speaker_encoder.parameters()},{'params':self.loss_fn.parameters()}]
        return torch.optim.RAdam(params,lr=self.lr,weight_decay=self.weight_decay)
     
def prepare_subfolder_fileset(subfolder,total_file_count,reqd_file_start_idx,reqd_file_end_idx):
    subfolder_fileset=[]
    for i in range(reqd_file_start_idx,reqd_file_end_idx):
        filename=f'{subfolder}/train-{i:05}-of-{total_file_count:05}.parquet'
        subfolder_fileset.append(filename)
        
    return subfolder_fileset


def get_indices(offset, NUM_EPOCHS, num_ds_split_parts=1, num_subfolders=30, num_filesets=4):
    # Calculate divisions
    epoch_size = num_ds_split_parts * num_subfolders * num_filesets
    ds_split_part_size = num_subfolders * num_filesets
    subfolder_idx_size = num_filesets

    # Compute each variable from the offset
    epoch = offset // epoch_size
    offset %= epoch_size
    
    ds_split_part = offset // ds_split_part_size
    offset %= ds_split_part_size
    
    subfolder_idx = offset // subfolder_idx_size
    offset %= subfolder_idx_size
    
    subfolder_fileset_idx = offset

    return epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx



        
    
if __name__ == "__main__":
    os.environ["HUGGINGFACE_TOKEN"] = "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"
    login(token="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")

    # kannada_raw_speech_meta_df = pd.read_excel('/kaggle/working/KannadaRawSpeechMetadata.xlsx')


    # kannada_raw_speech_meta_ds = Dataset.from_pandas(kannada_raw_speech_meta_df)

    # speakers = kannada_raw_speech_meta_ds['SpeakerID']

    # unique_speakers = set(kannada_raw_speech_meta_ds['SpeakerID'])

    # kannada_raw_speech_meta_ds = kannada_raw_speech_meta_ds.sort(['SpeakerID'])

    # speaker_count = Counter(speakers)

    # avg_audio_clip_per_speaker = sum(speaker_count.values())/len(speaker_count.keys())

    # min_audio_clip_per_speaker= min(speaker_count.values())

    # print(f'min_audio_clip_per_speaker:{min_audio_clip_per_speaker}')

    # max_audio_clip_per_speaker= max(speaker_count.values())

    # print(f'max_audio_clip_per_speaker:{max_audio_clip_per_speaker}')

    # remote_control = Studio(name="Capstone Model Training Arch Unet", teamspace="Vision-model", user="hemabhushanr3")

    # remote_control.start()

    extracted_metadata = pd.read_csv('/kaggle/working/SampleKnDataset/extracted_metadata.csv')

    extracted_metdata_ds = Dataset.from_pandas(extracted_metadata)

    folders = extracted_metadata['File Path'].to_list()

    unique_folders = set(map(lambda x:"/".join(x.split("/")[:6]),folders))

    folder_ds = Dataset.from_dict({"folder_path":list(unique_folders)})

    label_encoder = LabelEncoder()

    extracted_metdata_ds_label_encoded = label_encoder.fit_transform(extracted_metdata_ds['SpeakerID'])

    extracted_metdata_ds = extracted_metdata_ds.add_column('SpeakerLabel',extracted_metdata_ds_label_encoded)

    # extracted_metdata_ds = extracted_metdata_ds.train_test_split(test_size=0.15,seed=42)

    # extracted_metdata_ds_train = extracted_metdata_ds['train']
    
    extracted_data_len = len(extracted_metdata_ds)
    
    split_size = extracted_data_len//30
    
    
    
    NUM_SPEAKERS_PER_BATCH=4
    NUM_UTTERANCES_PER_SPEAKER=8
    NUM_EPOCHS=1
    # 65, 63, 66, 68, 58, 66, 62, 65, 66, 59, 59, 60, 68, 66, 65, 64, 60, 67, 63, 65, 62, 73, 59, 70, 65, 79, 83, 77, 84, 77, 1
    subfolder_parquet_count =  [65, 63, 66, 68, 58, 66, 62, 65, 66, 59, 59, 60, 68, 66, 65, 64, 60, 67, 63, 65, 62, 73, 59, 70, 65, 79, 83, 77, 84, 77]
    
    speaker_encoding_model = SpeakerEncodingModel(audio_config=BaseAudioConfig(),loss_name="angleproto",num_speaker_per_batch=NUM_SPEAKERS_PER_BATCH,num_utterances_per_speaker=NUM_UTTERANCES_PER_SPEAKER)
    
    curr_part=1
    
    curr_subfolder_fileset_kannada_speech_ds = None
    
    load_prev_epoch_from_hf=True
    
    file_path=None

    subfolder_split_count=4

    offset = 950

    prev_offset = offset-1

    epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx = get_indices(offset, NUM_EPOCHS,num_filesets=subfolder_split_count)

    if prev_offset>=0:
        prev_epoch, prev_ds_split_part, prev_subfolder_idx, prev_subfolder_fileset_idx = get_indices(prev_offset, NUM_EPOCHS,num_filesets=subfolder_split_count)

    
    # for epoch in range(NUM_EPOCHS):
    #    for ds_split_part in range(1):
    #        for subfolder_idx in range(1):
    subfolder=f'part_{ds_split_part+1}_sub_{subfolder_idx+1}'
    subfolder_split_size = subfolder_parquet_count[subfolder_idx]//subfolder_split_count
    curr_subfolder_samples_completed=0
    #for subfolder_fileset_idx in range(subfolder_split_count):
    reqd_file_start_idx = subfolder_fileset_idx*subfolder_split_size
    reqd_file_end_idx = reqd_file_start_idx+subfolder_split_size
    if reqd_file_end_idx>=subfolder_parquet_count[subfolder_idx] or subfolder_fileset_idx==3:
        reqd_file_end_idx = subfolder_parquet_count[subfolder_idx]
    curr_subfolder_file_set=prepare_subfolder_fileset(subfolder,subfolder_parquet_count[subfolder_idx],reqd_file_start_idx,reqd_file_end_idx)
    if curr_subfolder_fileset_kannada_speech_ds is not None:
    
        #!rm -r /kaggle/tmp/ds_cache
        #!rm -r /root/.cache
        
        curr_subfolder_fileset_kannada_speech_ds.cleanup_cache_files()
        subprocess.run(["rm","-rf","/root/.cache/ds_cache"])
        subprocess.run(["rm","-rf","/root/.cache/huggingface/hub"])
        #shutil.rmtree("/kaggle/working/ds_cache")
        #shutil.rmtree("/root/.cache/huggingface/hub")
        subprocess.run(["rm","-rf","/root/.cache/ds_cache"])
        subprocess.run(["rm","-rf","/root/.cache/huggingface/hub"])
        del curr_subfolder_fileset_kannada_speech_ds
        # subprocess.run(["rm","-r","/root/.cache"])
    curr_subfolder_fileset_kannada_speech_ds = load_dataset("Hemabhushan/kannada-speech",data_files=curr_subfolder_file_set,cache_dir="/root/.cache/ds_cache")
    
    curr_subfolder_fileset_kannada_speech_ds = curr_subfolder_fileset_kannada_speech_ds.with_format("torch")
    
    curr_subfolder_fileset_kannada_speech_ds = curr_subfolder_fileset_kannada_speech_ds['train']
    
    curr_ds_len = len(curr_subfolder_fileset_kannada_speech_ds)
    
    
    metadata_start_index = subfolder_idx*split_size
    
    metadata_end_index = metadata_start_index+split_size
    
    if metadata_end_index>=extracted_data_len:
        metadata_end_index = extracted_data_len
    
    curr_subfolder_metadata_ds = extracted_metdata_ds.select(range(metadata_start_index,metadata_end_index))
    
    curr_end = curr_subfolder_samples_completed+curr_ds_len
    
    if curr_end>=1143:
        curr_end=1143
    
    curr_subfolder_metadata_ds = curr_subfolder_metadata_ds.select(range(curr_subfolder_samples_completed,curr_end))
    
    #print('Curr Metadata Len',len(curr_subfolder_metadata_ds),curr_ds_len)
    
    curr_subfolder_samples_completed+=curr_ds_len
    
    training_dataloader = data.DataLoader(KannadaAudioDatasetHF(curr_subfolder_metadata_ds,curr_subfolder_fileset_kannada_speech_ds),sampler=SpeakerSampler(curr_subfolder_metadata_ds.to_pandas(),num_speakers=NUM_SPEAKERS_PER_BATCH, utterances_per_speaker=NUM_UTTERANCES_PER_SPEAKER,split="train"),batch_size=BATCH_SIZE,num_workers=DATALOADER_NUM_PROC,prefetch_factor=4,collate_fn=ds_collate_function)
    
    validation_dataloader = data.DataLoader(KannadaAudioDatasetHF(curr_subfolder_metadata_ds,curr_subfolder_fileset_kannada_speech_ds),sampler=SpeakerSampler(curr_subfolder_metadata_ds.to_pandas(),num_speakers=NUM_SPEAKERS_PER_BATCH, utterances_per_speaker=NUM_UTTERANCES_PER_SPEAKER,split="val"),batch_size=BATCH_SIZE,num_workers=DATALOADER_NUM_PROC,prefetch_factor=4,collate_fn=ds_collate_function)
    
    #ref_epochs = epoch*subfolder_split_count*len(subfolder_parquet_count)+subfolder_idx*subfolder_split_count+subfolder_fileset_idx+1

    ref_epochs =  offset+1
    
    logger = TensorBoardLogger(save_dir='/kaggle/working/tensorboard/', version=1, name="lightning_logs")
    
    trainer = L.Trainer(devices=1, accelerator="gpu", strategy="ddp",max_epochs=ref_epochs,logger=logger,num_sanity_val_steps=1,gradient_clip_val=3.0,use_distributed_sampler=False,log_every_n_steps=1)
    
    if load_prev_epoch_from_hf:
        model_repo_id="Hemabhushan/speaker-encoder-model"
    
        model_file_name=f'speaker_encoder_model_eph_{prev_epoch}_part_{prev_ds_split_part}_sub_{prev_subfolder_idx}_fileset_{prev_subfolder_fileset_idx}.bin'
    
        checkpoint_path = hf_hub_download(repo_id=model_repo_id, filename=model_file_name)
    
    if load_prev_epoch_from_hf:
        trainer.fit(speaker_encoding_model,training_dataloader,validation_dataloader,ckpt_path=checkpoint_path)                
    elif file_path is None:
        trainer.fit(speaker_encoding_model,training_dataloader,validation_dataloader)
    else:
        trainer.fit(speaker_encoding_model,training_dataloader,validation_dataloader,ckpt_path=file_path)
    
    file_path = '/kaggle/working/speaker_encoder.ckpt'
    
    trainer.save_checkpoint(file_path)

    model_repo_name = 'Hemabhushan/speaker-encoder-model'

    api = HfApi()

    # Upload the .pth file
    api.upload_file(
        path_or_fileobj=file_path, #epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx
        path_in_repo=f'speaker_encoder_model_eph_{epoch}_part_{ds_split_part}_sub_{subfolder_idx}_fileset_{subfolder_fileset_idx}.bin',  # This will be the file name in the repo
        repo_id=model_repo_name,
        token="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"
    )

