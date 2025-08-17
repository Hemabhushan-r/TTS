from datasets import Dataset,load_dataset
import os
# os.environ["LD_PRELOAD"] = 'libjemalloc.so.2'
from huggingface_hub import login
import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecision
import numpy as np
import torchaudio
import torchaudio.transforms as T
import json
from tqdm import tqdm
import lightning as L
from huggingface_hub import login,hf_hub_download,snapshot_download
from huggingface_hub import HfApi, HfFolder
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.plugins.precision import FSDPPrecision
from trainer.torch import NoamLR
from gpu_train import get_indices,prepare_subfolder_fileset,SpeakerEncodingModel
import gc
import tempfile
import textgrid
from dotenv import load_dotenv 

from torch.utils import data

import sys

load_dotenv()

TTS_PATH = os.getenv('/teamspace/studios/this_studio/TTS') or ''

sys.path.append(TTS_PATH)

os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'


from TTS.tts.utils.text.characters import BaseCharacters
from TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from TTS.tts.utils.text.characters import IPAPhonemes
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.utils.audio.processor import AudioProcessor
from TTS.tts.layers.tacotron.tacotron2 import Decoder, Encoder, Postnet
from TTS.config import load_config
from TTS.tts.layers.losses import MSELossMasked
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.layers.tacotron.attentions import MonotonicDynamicConvolutionAttention
from TTS.tts.layers.tacotron.tacotron2 import ConvBNBlock
from TTS.tts.utils.measures import alignment_diagonal_score

USE_TPU = False

def align_audio_in_memory(audio_tensor, sample_rate, transcription, dict_path, model_path):
    # Create temporary directories for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")
        text_path = os.path.join(temp_dir, "audio.txt")
        output_dir = os.path.join(temp_dir, "output")

        preproc_transcription=""

        for char in transcription:
            preproc_transcription+=f'{char} '

        with open('/kaggle/working/mfa_debug.txt','a') as f:
            f.write(f'{preproc_transcription}')
        
        # Save audio and transcription temporarily
        torchaudio.save(audio_path, audio_tensor, sample_rate)
        with open(text_path, 'w', encoding='utf-8') as file:
            file.write(preproc_transcription)
        os.makedirs(output_dir, exist_ok=True)

        # Perform alignment using MFA CLI through Python
        command = [
            "mfa", "align",
            temp_dir,  # Corpus directory
            dict_path,  # Dictionary path
            model_path,  # Acoustic model path
            output_dir  # Output directory
        ]
        os.system(" ".join(command))  # Run MFA alignment via CLI

        # Process the output TextGrid
        textgrid_path = os.path.join(output_dir, "audio.TextGrid")
        with open(textgrid_path, 'r') as tg_file:
            text_grid_result = tg_file.read()

        with open('/kaggle/working/grid_debug.txt','w') as f:
            f.write(text_grid_result)

        result = textgrid.TextGrid.fromFile(textgrid_path)
        
        return result  # Return the content of the TextGrid as a string


if USE_TPU:
    from torch_xla import runtime as xr
    import torch_xla.utils.utils as xu
    import torch_xla.distributed.spmd as xs
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.debug.profiler as xp



OUTPUTS_PER_STEP =2
SPEAKER_ENCODER_SAMPLE_RATE = 16000
SYNTHESIZER_SAMPLE_RATE = 22050
KN_DICT_PATH='/kaggle/working/Kannada_Dict.txt'
KN_AC_MODEL_PATH='/kaggle/working/Kannada_Acoustic_Model.zip'

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

if USE_TPU:
    xr.use_spmd()


def synthesizer_collate_fn(batch):
    b_transcription_ids,b_transcription_length,b_mel_spec,b_speaker_audio_resampled = zip(*batch)

    b_mel_lengths = [m.shape[1] for m in b_mel_spec]

    b_transcription_length = np.array([transcription_length for transcription_length in b_transcription_length])

    # compute 'stop token' targets
    b_stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in b_mel_lengths]

    # PAD stop targets
    b_stop_targets = prepare_stop_target(b_stop_targets,OUTPUTS_PER_STEP)

    # PAD sequences with longest instance in the batch
    b_transcription_ids = [np.array(transcription_ids) for transcription_ids in b_transcription_ids]
    b_transcription_ids = prepare_data(b_transcription_ids).astype(np.int32)

    # PAD features with longest instance
    # with open('dl_debug.txt','a') as f:
    #     f.write(f'In collate fn before {b_mel_spec[0].shape} ')
    b_mel_spec = prepare_tensor(b_mel_spec, OUTPUTS_PER_STEP)

    # b_linear_spec  = prepare_tensor(b_linear_spec, OUTPUTS_PER_STEP)

    # B x D x T --> B x T x D
    b_mel_spec = b_mel_spec.transpose(0, 2, 1)

    # b_linear_spec = b_linear_spec.transpose(0, 2, 1)

    # Find the maximum number of samples across all tensors
    max_num_samples = max(tensor.shape[1] for tensor in b_speaker_audio_resampled)

    # Pad each tensor along the num_samples dimension to match max_num_samples
    padded_tensors = [torch.nn.functional.pad(tensor, (0, max_num_samples - tensor.shape[1])) for tensor in b_speaker_audio_resampled]

    # Stack the padded tensors along a new dimension (e.g., batch dimension)
    b_speaker_audio_resampled = torch.stack(padded_tensors, dim=0)

    # convert things to pytorch
    b_transcription_length = torch.LongTensor(b_transcription_length)
    b_transcription_ids = torch.LongTensor(b_transcription_ids)
    b_mel_spec = torch.FloatTensor(b_mel_spec).contiguous()
    b_mel_lengths = torch.LongTensor(b_mel_lengths)
    b_stop_targets = torch.FloatTensor(b_stop_targets)
    # b_linear_spec = torch.FloatTensor(b_linear_spec).contiguous()

    # with open('dl_debug.txt','a') as f:
    #     f.write('In collate fn before return')

    return b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled

# MFA_FOLDER_PATH = snapshot_download("Hemabhushan/ksbvs-mfa-synthesizer",repo_type="dataset")

MFA_FOLDER_PATH = '/kaggle/input/kannada-speech-subset-mfa-textgrid'

# Define repository and local save path
repo_id = "Hemabhushan/ksbvs-mfa-synthesizer"  # Replace with the actual repo ID
local_dir = "./huggingface_repo"  # Change this to your preferred directory

# MFA_FOLDER_PATH = hf_hub_download(repo_id=repo_id,repo_type='dataset')

# Initialize Hugging Face API
# api = HfApi()

# List all files in the repository
# files = api.list_repo_files(repo_id=repo_id,repo_type='dataset')

# Download each file while preserving folder structure
# for file in files:
#     # Create local file path
#     local_file_path = os.path.join(local_dir, file)
    
#     # Ensure the directory structure exists
#     os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

#     # Download the file
#     downloaded_file = hf_hub_download(repo_id=repo_id,repo_type='dataset', filename=file, local_dir=local_dir)

#     print(f"Downloaded: {downloaded_file}")


SUBFOLDER_SPLIT_COUNT = 30


class SynthesizerDataset(data.IterableDataset):
    def __init__(
            self,
            ds_repo_id,
            ds_repo_subfolders,
            speech_ds_meta,
            file_to_idx_map,
            ipa_phonemes,
            tokenizer,
            audio_processor,
            speaker_encoder_sample_rate,
            synthesizer_sample_rate,
            offset):
        self.ds_repo_id = ds_repo_id
        self.ds_repo_subfolders = ds_repo_subfolders
        self.speech_ds_meta = speech_ds_meta
        self.file_to_idx_map = file_to_idx_map
        self.ipa_phonemes = ipa_phonemes
        self.tokenizer = tokenizer
        self.punct_trans_table = dict()
        for punct in self.tokenizer.characters.punctuations:
            self.punct_trans_table[punct]=""
        self.punct_trans_table = str.maketrans(self.punct_trans_table)
        self.audio_processor = audio_processor
        self.speaker_encoder_sample_rate = speaker_encoder_sample_rate
        self.synthesizer_sample_rate = synthesizer_sample_rate

        self.speaker_encoder_start_sec = 0
        self.speaker_encoder_duration_sec=15
        self.input_audio_sr = 48000
        self.offset = offset


    def get_iter(self):
        subfolder_parquet_count =  [65, 63, 66, 68, 58, 66, 62, 65, 66, 59, 59, 60, 68, 66, 65, 64, 60, 67, 63, 65, 62, 73, 59, 70, 65, 79, 83, 77, 84, 77]
        subfolder_split_count = SUBFOLDER_SPLIT_COUNT
        # for subfolder_idx,subfolder in enumerate(self.ds_repo_subfolders):
        for offset in [self.offset]:
            epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx = get_indices(offset,1,num_filesets=subfolder_split_count)

            subfolder=f'part_{ds_split_part+1}_sub_{subfolder_idx+1}'
            subfolder_split_size = subfolder_parquet_count[subfolder_idx]//subfolder_split_count
            curr_subfolder_samples_completed=0
            #for subfolder_fileset_idx in range(subfolder_split_count):
            reqd_file_start_idx = subfolder_fileset_idx*subfolder_split_size
            reqd_file_end_idx = reqd_file_start_idx+subfolder_split_size
            if reqd_file_end_idx>=subfolder_parquet_count[subfolder_idx] or subfolder_fileset_idx==subfolder_split_count-1:
                reqd_file_end_idx = subfolder_parquet_count[subfolder_idx]
            curr_subfolder_file_set=prepare_subfolder_fileset(subfolder,subfolder_parquet_count[subfolder_idx],reqd_file_start_idx,reqd_file_end_idx)


            with open('./offset_debug.txt','a') as f:
                f.write(f'offset {offset} curr_subfolder_file_set {curr_subfolder_file_set} \\n')
            
            # speech_ds = load_dataset(self.ds_repo_id,subfolder,streaming=True)
            speech_ds = load_dataset(self.ds_repo_id,data_files=curr_subfolder_file_set)

            speech_ds = speech_ds['train']

            speech_ds =  speech_ds.with_format("torch")

            sample_idx=0
            for sample in speech_ds:
                # file_path_split = sample['File Path'].chunk(0).take([0]).to_pylist()

                # file_path_split = file_path_split[0].split("/")

                # audio_sample_rate = sample['sample_rate'].chunk(0).take([0]).to_pylist()

                # audio_sample_rate = audio_sample_rate[0]
                
                # audio = sample['audio'].chunk(0).take([0]).to_pylist()

                # audio = audio[0]

                # audio = torch.tensor(audio)
                
                # with open('./dl_debug_arrow.txt','a') as f:
                #     f.write(f"file_path_split {file_path_split} audio shape {torch.tensor(audio).shape} len sample audio {len(sample['audio'])} len sample audio chunk {len(sample['audio'].chunk(0))}")

                file_path_split = sample['File Path'].split("/")
                file_name = f'{file_path_split[-6]}_{file_path_split[-1].replace(".wav","")}'

                transcription = self.speech_ds_meta[self.file_to_idx_map[file_name]]['transcription']

                audio = sample['audio']

                # with open('dl_debug.txt','a') as f:
                #     f.write(f'In dataloader audio dtype {audio.dtype}')

                audio_sample_rate = sample['sample_rate']

                
                synthesizer_audio_resampled = T.Resample(orig_freq=audio_sample_rate,new_freq=self.synthesizer_sample_rate)(audio).mean(dim=0, keepdim=True)
                
                skip_curr_sample=False
                try:
                    # MFA_FOLDER_PATH
                    textgrid_result = textgrid.TextGrid.fromFile(f'{MFA_FOLDER_PATH}/offset_{offset}/{file_name}.TextGrid')
                    # textgrid_result = align_audio_in_memory(synthesizer_audio_resampled,SYNTHESIZER_SAMPLE_RATE,transcription,KN_DICT_PATH,KN_AC_MODEL_PATH)
                except Exception as e:
                    print(f'Speech Alignment failed with error {e} skipping sample {file_name}')
                    skip_curr_sample=True


                if skip_curr_sample:
                    continue
                
                
                # with open('/kaggle/working/dl_debug.txt','a') as f:
                #         f.write(f'{textgrid_result}')
                
                
                aligned_text=""
                # punct_clean_transcription  = clean_transcription.translate(self.punct_trans_table).replace(" ","")
                clean_transcription  = transcription.strip()

                transcription_sentence_split = clean_transcription.split(".")

                sentence_boundaries = []  # To store end times of each sentence
                current_time = 0.0        # Tracks end time of each character
                audio_slices = []         # To store audio segments
                char_index = 0            # Index for alignment

                alignment_tiers = []

                for grid in textgrid_result:
                    for tier in grid:
                        if tier.mark not in ['<eps>','sil','spn','',' ']:
                            aligned_text+=f'{tier.mark}'
                            alignment_tiers.append(tier)
                            with open('/kaggle/working/t_bound_debug.txt','a') as f:
                                f.write(f'tier minTime {tier.minTime} maxTime {tier.maxTime} mark {tier.mark} end')
                            # if curr_trans_split_idx < len(transcription_sentence_split) and curr_trans_idx==len(transcription_sentence_split[curr_trans_split_idx]):
                            #     curr_trans_split_idx+=1
                            #     curr_trans_idx=0
                            # if curr_trans_split_idx < len(transcription_sentence_split) and curr_trans_idx<len(transcription_sentence_split[curr_trans_split_idx]) and transcription_sentence_split[curr_trans_split_idx][curr_trans_idx] in self.tokenizer.characters.punctuations:
                            #     curr_trans_idx+=1
                            

                            # if curr_trans_split_idx < len(transcription_sentence_split) and  curr_trans_idx<len(transcription_sentence_split[curr_trans_split_idx]) and transcription_sentence_split[curr_trans_split_idx][curr_trans_idx] in self.tokenizer.characters.punctuations:
                            #     curr_trans_idx+=1
                            # if curr_trans_split_idx < len(transcription_sentence_split) and tier.mark == transcription_sentence_split[curr_trans_split_idx][curr_trans_idx]:
                            #     curr_trans_idx+=1

                # with open('/kaggle/working/t_bound_debug.txt','a') as f:
                #     f.write(f'aligned text {aligned_text}')
                
                for char in clean_transcription:
                    if char == '.':
                        sentence_boundaries.append(current_time)
                    if char in self.tokenizer.characters.punctuations or char in [' ']:
                        continue
                    with open('/kaggle/working/new_char_debug.txt','a') as f:
                        f.write(f'char {char} mark {alignment_tiers[char_index].mark} end ')
                    if char_index < len(alignment_tiers) and alignment_tiers[char_index].mark == char:
                        # Get alignment timings for the current character
                        char_min_time, char_max_time = alignment_tiers[char_index].minTime,alignment_tiers[char_index].maxTime
                        current_time = char_max_time  # Track the end time of each character
                        char_index+=1
                        with open('/kaggle/working/t_bound_debug.txt','a') as f:
                            f.write(f'found {current_time}')

                audio_duration = len(synthesizer_audio_resampled[0])/self.synthesizer_sample_rate

                if current_time < audio_duration and len(transcription_sentence_split)>len(sentence_boundaries):
                    sentence_boundaries.append(current_time)
                    
                # Total frames in the audio data

                total_frames = audio.shape[-1]


                # Calculate frame offsets
                frame_offset = int(self.speaker_encoder_start_sec * self.input_audio_sr)
                num_frames = min(int(self.speaker_encoder_duration_sec * self.input_audio_sr), total_frames - frame_offset)
                
                # Extract the portion of the waveform
                audio_slice = audio[:, frame_offset:frame_offset + num_frames]


                speaker_audio_resampled = T.Resample(orig_freq=audio_sample_rate,new_freq=self.speaker_encoder_sample_rate)(audio_slice).mean(dim=0, keepdim=True)

                # with open('/kaggle/working/dl_debug.txt','a') as f:
                #         f.write(f'trans split {len(transcription_sentence_split)} sentence_boundaries {len(sentence_boundaries)}')
                
                # assert len(transcription_sentence_split) == len(sentence_boundaries)

                # print(len(transcription_sentence_split),len(sentence_boundaries))
                prev_time = 0.0
                for split_idx in range(len(transcription_sentence_split)):

                    if split_idx >= len(sentence_boundaries):
                        continue
                    synth_start_sample = int(prev_time * self.synthesizer_sample_rate)
                    synth_end_sample = int(sentence_boundaries[split_idx] * self.synthesizer_sample_rate)


                    prev_time = sentence_boundaries[split_idx]

                    curr_split_audio = synthesizer_audio_resampled[:, synth_start_sample:synth_end_sample]

                    curr_split_transcription = transcription_sentence_split[split_idx]

                    transcription_ids = self.tokenizer.text_to_ids(curr_split_transcription,language='kn')

                    transcription_length = len(transcription_ids)

                    with open('/kaggle/working/dl_debug.txt','a') as f:
                        f.write('before mel spec obtained')

                    if len(curr_split_audio.squeeze()) == 0:
                        continue
                
                    mel_spec = self.audio_processor.melspectrogram(curr_split_audio.squeeze().numpy())

                    # linear_spec = self.audio_processor.spectrogram(synthesizer_audio_resampled.squeeze().numpy())

                    with open('/kaggle/working/dl_debug.txt','a') as f:
                        f.write('mel spec obtained')
                    
                    if mel_spec.shape[1]>2500:
                        skip_yield=True
                    else:
                        skip_yield=False

                    sample_idx+=1
                    gc.collect()
                    if not skip_yield:
                        yield transcription_ids,transcription_length,mel_spec,speaker_audio_resampled


    def __iter__(self):
        return self.get_iter()


class SynthesizerModel(L.LightningModule):
    def __init__(self,tacotron2_config,speaker_encoder_ckpt_offset,speaker_encoder_subfolder_split_count=4,speaker_enc_model_repo_id="Hemabhushan/speaker-encoder-model",source_sample_rate=48000):
        super().__init__()
        self.text_to_mel_model =None
        self.speaker_encoder = None
        self.source_sample_rate = source_sample_rate
        self.tacotron2_config = tacotron2_config
        self.speaker_enc_model_repo_id =  speaker_enc_model_repo_id
        self.speaker_encoder_ckpt_offset = speaker_encoder_ckpt_offset
        self.speaker_encoder_subfolder_split_count = speaker_encoder_subfolder_split_count
        self.lr = self.tacotron2_config.lr
        self.betas = self.tacotron2_config.optimizer_params['betas']
        self.weight_decay = self.tacotron2_config.optimizer_params['weight_decay']
        self.lr_scheduler_warmup_steps = self.tacotron2_config.lr_scheduler_params['warmup_steps']
        # self.criterion = MSELossMasked(seq_len_norm=False)
        # self.criterion_st = nn.BCEWithLogitsLoss()

        self.criterion = None

    def configure_model(self):
        self.text_to_mel_model = Tacotron2(self.tacotron2_config)

        epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx = get_indices(self.speaker_encoder_ckpt_offset,1,num_filesets=self.speaker_encoder_subfolder_split_count)
    
        model_file_name=f'speaker_encoder_model_eph_{epoch}_part_{ds_split_part}_sub_{subfolder_idx}_fileset_{subfolder_fileset_idx}.bin'
    
        checkpoint_path = hf_hub_download(repo_id=self.speaker_enc_model_repo_id, filename=model_file_name)
        
        self.speaker_encoder = SpeakerEncodingModel.load_from_checkpoint(checkpoint_path)

        self.speaker_encoder.eval()
        
        for name,param in self.speaker_encoder.named_parameters():
            param.requires_grad = False

        if self.criterion is None:
            self.criterion = self.text_to_mel_model.get_criterion()

    def training_step(self,batch,batch_idx):
        
        b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = batch

        print(batch[0].shape,batch[1].shape,batch[2].shape,batch[3].shape,batch[4].shape,batch[5].shape)

        b_stop_targets = b_stop_targets.view(b_transcription_ids.shape[0], b_stop_targets.size(1) // self.tacotron2_config.r, -1)
        
        b_stop_targets = (b_stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()
        
        # b_mel_postnet_spec = b_mel_spec.detach().clone()

        if b_mel_lengths.max() % self.text_to_mel_model.decoder.r != 0:
            alignment_lengths = (
                b_mel_lengths + (self.text_to_mel_model.decoder.r - (b_mel_lengths.max() % self.text_to_mel_model.decoder.r))
            ) // self.text_to_mel_model.decoder.r
        else:
            alignment_lengths = b_mel_lengths // self.text_to_mel_model.decoder.r

        alignment_lengths = alignment_lengths.to(self.device)

        stop_target_lengths = torch.divide(b_mel_lengths, self.tacotron2_config.r).ceil_().to(self.device)

        # with open('/kaggle/working/float_test.txt','w') as f:
        #     f.write(f'b_transcription_ids {b_transcription_ids.dtype} b_transcription_length {b_transcription_length.dtype} b_mel_spec {b_mel_spec.dtype} b_mel_lengths {b_mel_lengths.dtype} b_speaker_audio_resampled {b_speaker_audio_resampled.dtype} alignment_lengths {alignment_lengths.dtype} stop_target_lengths {stop_target_lengths.dtype} b_stop_targets {b_stop_targets.dtype}')
        
        # with open('/kaggle/working/float_test.txt','a') as f:
        #     f.write('self.speaker_encoder.lstm_speaker_encoder being entered')
        
        with torch.no_grad():
            b_speaker_audio_resampled = b_speaker_audio_resampled.flatten(0,1)

            # b_speaker_audio_resampled = b_speaker_audio_resampled.to("cpu")

            # self.speaker_encoder.lstm_speaker_encoder =  self.speaker_encoder.lstm_speaker_encoder.to("cpu")
        
            b_speaker_embeddings = self.speaker_encoder.lstm_speaker_encoder(b_speaker_audio_resampled)

        b_speaker_embeddings = b_speaker_embeddings.to(b_transcription_ids.device)
        
        # with open('/kaggle/working/float_test.txt','w') as f:
        #     f.write('self.speaker_encoder.lstm_speaker_encoder crossed')
        
        outputs = self.text_to_mel_model.forward(
                b_transcription_ids, b_transcription_length, b_mel_spec, b_mel_lengths, aux_input={"d_vectors": b_speaker_embeddings}
            ) 

        # with open('/kaggle/working/float_test.txt','a') as f:
        #     f.write('self.text_to_mel_model.forward crossed')

        file_op = f'outputs {outputs} \\n'

        # loss = self.criterion(outputs["decoder_outputs"], b_mel_spec, b_mel_lengths)

        loss_dict = self.criterion(
                outputs["model_outputs"],
                outputs["decoder_outputs"],
                b_mel_spec,
                None,
                outputs["stop_tokens"],
                b_stop_targets,
                stop_target_lengths,
                outputs["capacitron_vae_outputs"] if self.text_to_mel_model.capacitron_vae else None,
                b_mel_lengths,
                None if outputs["decoder_outputs_backward"] is None else outputs["decoder_outputs_backward"],
                outputs["alignments"],
                alignment_lengths,
                None if outputs["alignments_backward"] is None else outputs["alignments_backward"],
                b_transcription_length,
            )

        # with open('/kaggle/working/float_test.txt','a') as f:
        #     f.write('self.criterion crossed')


        file_op += f'loss {loss_dict} \\n'
        
        # stop_loss = self.criterion_st(outputs["stop_tokens"], b_stop_targets)

        # file_op += f'stop_loss {stop_loss} \\n'
        
        # loss = loss + self.criterion(outputs["model_outputs"], b_linear_spec, b_mel_lengths) + stop_loss

        # file_op += f'loss {loss} \\n'

        # with open('/kaggle/working/file_op.txt','w') as f:
        #     f.write(file_op)

        align_error = 1 - alignment_diagonal_score(outputs["alignments"])
        loss_dict["align_error"] = align_error

        for key in loss_dict.keys():
            self.log(f'Training {key}', loss_dict[key], on_step=True, on_epoch=True, logger=True)
        

        # self.log('Training Loss', loss.detach().item(), on_step=True, on_epoch=True, logger=True)
        
        loss = loss_dict["loss"]
        
        return loss

    def validation_step(self,batch,batch_idx):
        
        b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = batch

        print(batch[0].shape,batch[1].shape,batch[2].shape,batch[3].shape,batch[4].shape,batch[5].shape)

        b_stop_targets = b_stop_targets.view(b_transcription_ids.shape[0], b_stop_targets.size(1) // self.tacotron2_config.r, -1)
        
        b_stop_targets = (b_stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze()
        
        
        # b_mel_postnet_spec = b_mel_spec.detach().clone()

        if b_mel_lengths.max() % self.text_to_mel_model.decoder.r != 0:
            alignment_lengths = (
                b_mel_lengths + (self.text_to_mel_model.decoder.r - (b_mel_lengths.max() % self.text_to_mel_model.decoder.r))
            ) // self.text_to_mel_model.decoder.r
        else:
            alignment_lengths = b_mel_lengths // self.text_to_mel_model.decoder.r

        alignment_lengths = alignment_lengths.to(self.device)

        stop_target_lengths = torch.divide(b_mel_lengths, self.tacotron2_config.r).ceil_().to(self.device)

        
        with torch.no_grad():
            b_speaker_audio_resampled = b_speaker_audio_resampled.flatten(0,1)
            
            b_speaker_embeddings = self.speaker_encoder.lstm_speaker_encoder(b_speaker_audio_resampled)

        outputs = self.text_to_mel_model.forward(
                b_transcription_ids, b_transcription_length, b_mel_spec, b_mel_lengths, aux_input={"d_vectors": b_speaker_embeddings}
            )

        loss_dict = self.criterion(
                outputs["model_outputs"],
                outputs["decoder_outputs"],
                b_mel_spec,
                None,
                outputs["stop_tokens"],
                b_stop_targets,
                stop_target_lengths,
                outputs["capacitron_vae_outputs"] if self.text_to_mel_model.capacitron_vae else None,
                b_mel_lengths,
                None if outputs["decoder_outputs_backward"] is None else outputs["decoder_outputs_backward"],
                outputs["alignments"],
                alignment_lengths,
                None if outputs["alignments_backward"] is None else outputs["alignments_backward"],
                b_transcription_length,
            )

        # loss = self.criterion(outputs["decoder_outputs"], b_mel_spec, b_mel_lengths)
        
        # stop_loss = self.criterion_st(outputs["stop_tokens"], b_stop_targets)
        
        # loss = loss + self.criterion(outputs["model_outputs"], b_linear_spec, b_mel_lengths) + stop_loss

        align_error = 1 - alignment_diagonal_score(outputs["alignments"])
        loss_dict["align_error"] = align_error
        
        for key in loss_dict.keys():
            self.log(f'Validation {key}', loss_dict[key], on_step=True, on_epoch=True, logger=True)
        
        
        # self.log('Validation Loss', loss.detach().item(), on_step=True, on_epoch=True, logger=True)

        loss = loss_dict["loss"]
        
        return loss

    def configure_optimizers(self):
        params = self.text_to_mel_model.parameters()
        optimizer = torch.optim.RAdam(params,lr=self.lr,betas=tuple(self.betas),weight_decay=self.weight_decay,eps=1e-06)
        scheduler = NoamLR(optimizer,warmup_steps=self.lr_scheduler_warmup_steps)
        return {"optimizer":optimizer,"lr_scheduler":scheduler}
        

 


if __name__ == "__main__":
    os.environ["HUGGINGFACE_TOKEN"] = "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"
    login(token="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")

    # extracted_metadata_ds = load_dataset("Hemabhushan/kannada-speech","part_1_text")

    # extracted_metadata_ds = extracted_metadata_ds['train']

    ds_repo_id = "Hemabhushan/kannada-speech"

    ds_meta_subfolder = "part_1_text"

    speech_ds_meta = load_dataset(ds_repo_id,ds_meta_subfolder)

    speech_ds_meta = speech_ds_meta['train']

    file_to_idx_map={}

    for sample_idx,sample in enumerate(speech_ds_meta):
        file_path_split = sample['File Path'].split("/")
        file_name = f'{file_path_split[-6]}_{file_path_split[-1].replace(".wav","")}'
        # print(f'sample_idx {sample_idx}\t {file_name} \t file path {sample["File Path"]}')
        if file_name in file_to_idx_map.keys():
            print(file_name)
            break
        file_to_idx_map[file_name] = sample_idx
        # print(sample['File Path'])
        

    print(len(file_to_idx_map.keys()),len(speech_ds_meta))
    
    assert len(file_to_idx_map.keys()) == len(speech_ds_meta)

    

    with open('/kaggle/working/custom_tts_config.json','r') as f:
        tacotron_config_json = json.loads(f.read()) 

    ipa_phonemes = IPAPhonemes()

    config = load_config('/kaggle/working/custom_tts_config.json')


    config.d_vector_dim = 256 # Very Imp

    tacotron_config = Tacotron2Config(**config)

    # print(tacotron_config.characters)

    audio_processor = AudioProcessor.init_from_config(tacotron_config)


    tokenizer,config = TTSTokenizer.init_from_config(tacotron_config,ipa_phonemes)

    ipa_phonemes.characters+="ʰ"

    ipa_phonemes.punctuations= ipa_phonemes.punctuations[:-1]+"“”"+ipa_phonemes.punctuations[-1]

    tokenizer.characters = ipa_phonemes

    ds_repo_subfolders = []

    part = 1

    for i in range(30):
        subfolder_name = f'part_{part}_sub_{i+1}'
        ds_repo_subfolders.append(subfolder_name)

    total_subfolder_count = len(ds_repo_subfolders)

    validation_subfolder_count = 6

    if USE_TPU:
        server = xp.start_server(9012)

        profile_logdir='/kaggle/working/tensorboard/'
        
        device = torch_xla.device()
        num_devices = xr.global_runtime_device_count()
        device_ids = np.arange(num_devices)
        mesh_shape = (num_devices,)
        mesh = xs.Mesh(device_ids, mesh_shape, ('data',))

        num_epochs = 1

        ref_epochs = 0

        BATCH_SIZE = 1

        DATALOADER_WORKERS = 4

        DTYPE_PT = torch.float32

        offset = 0

        training_data_loader = data.DataLoader(SynthesizerDataset(
            ds_repo_id,
            ds_repo_subfolders[:total_subfolder_count-validation_subfolder_count],
            speech_ds_meta,
            file_to_idx_map,
            ipa_phonemes,
            tokenizer,
            audio_processor,
            SPEAKER_ENCODER_SAMPLE_RATE,
            SYNTHESIZER_SAMPLE_RATE,offset=0),batch_size=BATCH_SIZE,num_workers=3,collate_fn=synthesizer_collate_fn,prefetch_factor=16)    

        val_offset = (total_subfolder_count-validation_subfolder_count+(offset%6))*20
    
        validation_data_loader = data.DataLoader(SynthesizerDataset(
            ds_repo_id,
            ds_repo_subfolders[-validation_subfolder_count:],
            speech_ds_meta,
            file_to_idx_map,
            ipa_phonemes,
            tokenizer,
            audio_processor,
            SPEAKER_ENCODER_SAMPLE_RATE,
            SYNTHESIZER_SAMPLE_RATE,offset=val_offset),batch_size=BATCH_SIZE,num_workers=3,collate_fn=synthesizer_collate_fn,prefetch_factor=16)    
    
        
        speaker_encoder_ckpt_offset  = 1200

        tacotron_config.num_chars = ipa_phonemes.num_chars
    
        print(f'Vocab num_chars: {tacotron_config.num_chars}')
    
        synthesizer_model = SynthesizerModel(tacotron_config,speaker_encoder_ckpt_offset)


        synthesizer_model.configure_model()        

        synthesizer_model = synthesizer_model.to(device)

        optimizer_lr_scheduler_dict = synthesizer_model.configure_optimizers()

        optimizer = optimizer_lr_scheduler_dict['optimizer']

        lr_scheduler = optimizer_lr_scheduler_dict['lr_scheduler']

        writer = SummaryWriter('/kaggle/working/tensorboard/')

        train_loader_len = (1143*len(ds_repo_subfolders[:total_subfolder_count-validation_subfolder_count]))//BATCH_SIZE

        validation_loader_len = (1143*len(ds_repo_subfolders[-validation_subfolder_count:]))//BATCH_SIZE
        
        if ref_epochs>0:
            model_training_continued = True
        else:
            model_training_continued = False


        if model_training_continued:
            model_repo_id = 'Hemabhushan/text-to-mel-model'
            model_file_name = f'text_to_mel_model_kn_split_speaker_enc_offset_{speaker_encoder_ckpt_offset}_eph_{ref_epochs-1}.bin'
            checkpoint_path = hf_hub_download(repo_id=model_repo_id, filename=model_file_name)
            model_train_state_dict = torch.load(checkpoint_path)
            synthesizer_model.load_state_dict(model_train_state_dict['model'])
            optimizer.load_state_dict(model_train_state_dict['optimizer'])
            lr_scheduler.load_state_dict(model_train_state_dict['lr_scheduler'])
        
        
        print(f'Train Loader Len: {train_loader_len}')
        print(f'Validation Loader Len: {validation_loader_len}')

        for epoch in tqdm(range(num_epochs)):
            ## Training Loop
            batch_idx=0
            synthesizer_model.text_to_mel_model.train()
            for batch in tqdm(training_data_loader,total=train_loader_len):
                with xp.StepTrace('Training_step', step_num=batch_idx): 
                    if epoch==0 and (batch_idx==10 or batch_idx==12):
                        try:
                            xp.trace_detached('localhost:9012', profile_logdir)
                        except Exception as e:
                            print(f'Profiling failed with exception {e}')
                    b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = batch

                    # b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = b_transcription_ids.to(DTYPE_PT),b_transcription_length.to(DTYPE_PT),b_mel_spec.to(DTYPE_PT),b_mel_lengths.to(DTYPE_PT),b_stop_targets.to(DTYPE_PT),b_speaker_audio_resampled.to(DTYPE_PT)

                    b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = b_transcription_ids.to(device),b_transcription_length.to(device),b_mel_spec.to(device),b_mel_lengths.to(device),b_stop_targets.to(device),b_speaker_audio_resampled.to(device)
                    
                    
                    xs.mark_sharding(b_transcription_ids,mesh,('data',None))
                    xs.mark_sharding(b_transcription_length,mesh,('data',))
                    xs.mark_sharding(b_mel_spec,mesh,('data',None,None))
                    xs.mark_sharding(b_mel_lengths,mesh,('data',))
                    xs.mark_sharding(b_stop_targets,mesh,('data',None))
                    xs.mark_sharding(b_speaker_audio_resampled,mesh,('data',None,None))

                    batch = b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled 

                    
                    optimizer.zero_grad()
                
                    loss = synthesizer_model.training_step(batch,batch_idx)
    
                    loss.backward()
    
                    optimizer.step()

                    lr_scheduler.step()

                    writer.add_scalar("Training Loss",loss.detach().item(),epoch*train_loader_len+batch_idx)
                

                xm.mark_step()

                batch_idx+=1

            
            try:
                file_path = f"/kaggle/working/text_to_mel_model_state.pth"
                
                xm.save({"epoch":epoch,"model":synthesizer_model.state_dict(),"optimizer":optimizer.state_dict(),"lr_scheduler":lr_scheduler.state_dict()},file_path)
                
                model_repo_name = 'Hemabhushan/text-to-mel-model'
        
                api = HfApi()
            
                # Upload the .pth file
                api.upload_file(
                    path_or_fileobj=file_path, #epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx
                    path_in_repo=f'text_to_mel_model_kn_split_speaker_enc_offset_{speaker_encoder_ckpt_offset}_eph_{ref_epochs+epoch}.bin',  # This will be the file name in the repo
                    repo_id=model_repo_name,
                    token="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"
                )
            except Exception as e:
                print('Failed uploading Model with exception {e}')

            ## Validation Loop

            batch_idx=0
            synthesizer_model.text_to_mel_model.eval()

            with torch.no_grad():
                for batch in tqdm(validation_data_loader,total=validation_loader_len):
                    with xp.StepTrace('Validation_step', step_num=batch_idx): 
                        
                        
                        b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = batch

                        # b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = b_transcription_ids.to(DTYPE_PT),b_transcription_length.to(DTYPE_PT),b_mel_spec.to(DTYPE_PT),b_mel_lengths.to(DTYPE_PT),b_stop_targets.to(DTYPE_PT),b_speaker_audio_resampled.to(DTYPE_PT)

                        b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled = b_transcription_ids.to(device),b_transcription_length.to(device),b_mel_spec.to(device),b_mel_lengths.to(device),b_stop_targets.to(device),b_speaker_audio_resampled.to(device)

                        xs.mark_sharding(b_transcription_ids,mesh,('data',None))
                        xs.mark_sharding(b_transcription_length,mesh,('data',))
                        xs.mark_sharding(b_mel_spec,mesh,('data',None,None))
                        xs.mark_sharding(b_mel_lengths,mesh,('data',))
                        xs.mark_sharding(b_stop_targets,mesh,('data',None))
                        xs.mark_sharding(b_speaker_audio_resampled,mesh,('data',None,None))

                        batch = b_transcription_ids,b_transcription_length,b_mel_spec,b_mel_lengths,b_stop_targets,b_speaker_audio_resampled 

                        
                        loss = synthesizer_model.validation_step(batch,batch_idx)

                        writer.add_scalar("Validation Loss",loss.detach().item(),epoch*validation_loader_len+batch_idx)
                
                    batch_idx+=1
                        
    else:

        
       
    
        logger = TensorBoardLogger(save_dir='/kaggle/working/tensorboard/', version=1, name="lightning_logs")


        
        activation_checkpointing_policy={
             # MonotonicDynamicConvolutionAttention,
             ConvBNBlock,
        },
        strategy = FSDPStrategy(
        # Enable activation checkpointing on these layers
        precision_plugin = FSDPPrecision("16-true"),  
        # cpu_offload=True,
        # activation_checkpointing_policy={
        #      # MonotonicDynamicConvolutionAttention,
        #      ConvBNBlock,
        #      nn.Linear,
        # },
        # mixed_precision=MixedPrecision(param_dtype=torch.float16) # , cast_forward_inputs=True
    )
        
        curr_start_offset_var=2976
        # Rem Change Failed Offsets:25,26,30,45,53,65,75,81,84,89,90

        curr_start_offset = curr_start_offset_var
        
        FAILED_OFFSETS = [25, 26, 30, 45, 53,54,55, 65, 75, 81, 84, 89, 90] # 134-136
        while (curr_start_offset%90) in FAILED_OFFSETS: 
            curr_start_offset+=1
        
        ref_epochs = curr_start_offset + 1 # //90 and 
        
        speaker_encoder_ckpt_offset  = 1468
    
        if ref_epochs>1:
            load_prev_epoch_from_hf=True
            prev_offset = curr_start_offset-1
            while (prev_offset%90) in FAILED_OFFSETS:
                prev_offset-=1 # num_subfolders=3,
            prev_epoch, prev_ds_split_part, prev_subfolder_idx, prev_subfolder_fileset_idx = get_indices(prev_offset,1,num_subfolders=3,num_filesets=SUBFOLDER_SPLIT_COUNT)
        else:
            load_prev_epoch_from_hf=False

        
        
        if load_prev_epoch_from_hf:
            model_repo_id="Hemabhushan/text-to-mel-model"
        
            model_file_name=f'text_to_mel_model_kn_split_speaker_enc_offset_{speaker_encoder_ckpt_offset}_eph_{prev_epoch}_ds_split_part_{prev_ds_split_part}_subfolder_{prev_subfolder_idx}_fileset_{prev_subfolder_fileset_idx}.bin'
        
            checkpoint_path = hf_hub_download(repo_id=model_repo_id, filename=model_file_name)
    
        
        tacotron_config.num_chars = ipa_phonemes.num_chars
    
        print(f'Vocab num_chars: {tacotron_config.num_chars}')
    
        synthesizer_model = SynthesizerModel(tacotron_config,speaker_encoder_ckpt_offset)

        offset_per_run = 1

        last_completed_offset=0

        file_path = None
        
        epoch_offset = curr_start_offset

        offset = curr_start_offset%90

        curr_start_offset = curr_start_offset%90

        

        

        for offset in range(curr_start_offset,curr_start_offset+offset_per_run):
            if offset in [25,26,30,45,53,65,75,81,84,89,90]:
                continue
            strategy = FSDPStrategy(
                # Enable activation checkpointing on these layers
                precision_plugin = FSDPPrecision("16-true"),  
                # cpu_offload=True,
                # activation_checkpointing_policy={
                #      # MonotonicDynamicConvolutionAttention,
                #      ConvBNBlock,
                #      nn.Linear,
                # },
                # mixed_precision=MixedPrecision(param_dtype=torch.float16) # , cast_forward_inputs=True
            )
            # gradient_clip_algorithm="value", 
            trainer = L.Trainer(devices=1,max_epochs=epoch_offset+1,logger=logger,num_sanity_val_steps=0,gradient_clip_val=5,use_distributed_sampler=False,log_every_n_steps=1)
            # trainer = L.Trainer(accelerator="cpu",max_epochs=offset+1,logger=logger,num_sanity_val_steps=0,gradient_clip_val=0.05,gradient_clip_algorithm="value",use_distributed_sampler=False,log_every_n_steps=1)
            
            training_data_loader = data.DataLoader(SynthesizerDataset(
                ds_repo_id,
                ds_repo_subfolders[:total_subfolder_count-validation_subfolder_count],
                speech_ds_meta,
                file_to_idx_map,
                ipa_phonemes,
                tokenizer,
                audio_processor,
                SPEAKER_ENCODER_SAMPLE_RATE,
                SYNTHESIZER_SAMPLE_RATE,offset=offset),batch_size=16, drop_last=True,num_workers=1,collate_fn=synthesizer_collate_fn,prefetch_factor=1)    

            val_offset = offset+1 if offset%6==4 else offset
            
            val_offset = (total_subfolder_count-validation_subfolder_count+(val_offset%6))*SUBFOLDER_SPLIT_COUNT
            
            validation_data_loader = data.DataLoader(SynthesizerDataset(
                ds_repo_id,
                ds_repo_subfolders[-validation_subfolder_count:],
                speech_ds_meta,
                file_to_idx_map,
                ipa_phonemes,
                tokenizer,
                audio_processor,
                SPEAKER_ENCODER_SAMPLE_RATE,
                SYNTHESIZER_SAMPLE_RATE,offset=val_offset),batch_size=16,drop_last=True,num_workers=1,collate_fn=synthesizer_collate_fn,prefetch_factor=1)    
        
        
    
    
    
    
            
            if offset > curr_start_offset:
                trainer.fit(synthesizer_model,training_data_loader,validation_data_loader,ckpt_path=file_path)               
            elif load_prev_epoch_from_hf:
                trainer.fit(synthesizer_model,training_data_loader,validation_data_loader,ckpt_path=checkpoint_path)
            else:
                trainer.fit(synthesizer_model,training_data_loader,validation_data_loader)
            
        
            file_path = '/kaggle/working/text_to_mel_model.ckpt'
            
            trainer.save_checkpoint(file_path)
    
            last_completed_offset = epoch_offset

    
            model_repo_name = 'Hemabhushan/text-to-mel-model'
        
            api = HfApi()
    
            epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx = get_indices(last_completed_offset,1,num_subfolders=3,num_filesets=SUBFOLDER_SPLIT_COUNT)
            
        
            # Upload the .pth file
            api.upload_file(
                path_or_fileobj=file_path, #epoch, ds_split_part, subfolder_idx, subfolder_fileset_idx
                path_in_repo=f'text_to_mel_model_kn_split_speaker_enc_offset_{speaker_encoder_ckpt_offset}_eph_{epoch}_ds_split_part_{ds_split_part}_subfolder_{subfolder_idx}_fileset_{subfolder_fileset_idx}.bin',  # This will be the file name in the repo
                repo_id=model_repo_name,
                token="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"
            )

            epoch_offset+=1


    # for batch_idx,batch in enumerate(synthesizer_data_loader):
    #     print(batch[0].shape,batch[1].shape,batch[2].shape,batch[3].shape,batch[4].shape,batch[5].shape)

    #     if batch_idx == 2:
    #         breakV307 Offset 184-189

