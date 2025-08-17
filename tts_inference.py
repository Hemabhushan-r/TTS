import io
import os
from huggingface_hub import login,hf_hub_download
from datasets import load_dataset
import json




from gpu_train import get_indices 
from text_to_model_train import SynthesizerModel


# from TTS.config import load_config
# from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.utils.text.characters import IPAPhonemes
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.utils.audio.processor import AudioProcessor
from TTS.tts.utils.managers import EmbeddingManager
from TTS.config import load_config
from TTS.config import BaseAudioConfig
from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig



#Download models

os.environ["HUGGINGFACE_TOKEN"] = "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"
login(token="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")


NUM_SPEAKERS_PER_BATCH=4
NUM_UTTERANCES_PER_SPEAKER=8
NUM_EPOCHS=1

subfolder_split_count=4

speaker_encoder_ckpt_offset  = 1468

text_to_mel_model_ckpt_offset = 3060

offset = 950

curr_offset = offset-1

SUBFOLDER_SPLIT_COUNT = 30

curr_epoch, curr_ds_split_part, curr_subfolder_idx, curr_subfolder_fileset_idx = get_indices(text_to_mel_model_ckpt_offset,1,num_subfolders=3,num_filesets=SUBFOLDER_SPLIT_COUNT)


model_repo_id="Hemabhushan/text-to-mel-model"
        
model_file_name=f'text_to_mel_model_kn_split_speaker_enc_offset_{speaker_encoder_ckpt_offset}_eph_{curr_epoch}_ds_split_part_{curr_ds_split_part}_subfolder_{curr_subfolder_idx}_fileset_{curr_subfolder_fileset_idx}.bin'

text_to_mel_model_checkpoint_path = hf_hub_download(repo_id=model_repo_id, filename=model_file_name)

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



with open('/teamspace/studios/this_studio/TTS/custom_tts_config.json','r') as f:
    tacotron_config_json = json.loads(f.read()) 

ipa_phonemes = IPAPhonemes()

config = load_config('/teamspace/studios/this_studio/TTS/custom_tts_config.json')


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


tacotron_config.num_chars = ipa_phonemes.num_chars
    
print(f'Vocab num_chars: {tacotron_config.num_chars}')

# synthesizer_model = SynthesizerModel(tacotron_config,speaker_encoder_ckpt_offset)

synthesizer_model = SynthesizerModel.load_from_checkpoint(text_to_mel_model_checkpoint_path,tacotron2_config=tacotron_config,
    speaker_encoder_ckpt_offset=speaker_encoder_ckpt_offset)

# synthesizer_model.configure_model()

vocoder_path  = '/teamspace/studios/this_studio/.local/share/tts/vocoder_models--en--ljspeech--multiband-melgan/model_file.pth'

vocoder_config_path = '/teamspace/studios/this_studio/.local/share/tts/vocoder_models--en--ljspeech--multiband-melgan/config.json'

tts_model_path = ''

tts_config_path = '/teamspace/studios/this_studio/TTS/custom_tts_config.json'

speaker_encoder_audio_config = BaseAudioConfig()

# load models
synthesizer = Synthesizer(
    tts_checkpoint=tts_model_path,
    tts_config_path=tts_config_path,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=False,
)



synthesizer.tts_model = synthesizer_model.text_to_mel_model
synthesizer.tts_model.speaker_manager = EmbeddingManager()
synthesizer.tts_model.tokenizer = tokenizer
synthesizer.tts_model.speaker_manager.encoder = synthesizer_model.speaker_encoder.lstm_speaker_encoder
synthesizer.tts_model.speaker_manager.encoder_config = SpeakerEncoderConfig()
synthesizer.tts_model.speaker_manager.encoder_config.model_params['use_torch_spec'] = True 
synthesizer.tts_model.speaker_manager.encoder_ap =  AudioProcessor(**speaker_encoder_audio_config)
synthesizer.tts_config = load_config(tts_config_path)
synthesizer.tts_model.num_speakers = 2
synthesizer.tts_model.ap = AudioProcessor(**synthesizer.tts_config.audio)

 
if __name__ == '__main__':
    text_arr = [
        'ಸರ್ವೇಶ್ವರ್ ದಯಾಲ್ ಸಕ್ಸೇನಾ ಸರ್ವೇಶ್ವರ್ ದಯಾಲ್ ಸಕ್ಸೇನಾ ಸರ್ವೇಶ್ವರ್ ದಯಾಲ್ ಸಕ್ಸೇನಾ',
        'ಮನುಷ್ಯನನ್ನು ಪರಿಪೂರ್ಣನನ್ನಾಗಿ ರೂಪಿಸುವುದು ಅವನ ಉನ್ನತವಾದ ಚಿಂತನೆ. ತನ್ನ ಬದುಕನ್ನು ರೂಪಿಸಿಕೊಳ್ಳುವವನು ತಾನೇ ಎಂಬದನ್ನು ಮೊದಲು ಆತ ತಿಳಿದಿರಬೇಕು. ಏಕೆಂದರೆ ಆತನ ಒಳ್ಳೆಯ ಆಲೋಚನೆಗಳು ಆತನನ್ನು ಒಳ್ಳೆಯವನನ್ನಾಗಿ ರೂಪಿಸುತ್ತವೆ. ಆತನ ಚಿಂತನೆಗಳು ಕೆಟ್ಟದಾಗಿದ್ದರೆ ಆತನನ್ನು ವಿನಾಶದ ಅಂಚಿಗೊಯ್ಯತ್ತವೆ. ಆದ್ದರಿಂದ ಮನುಷ್ಯನ ವಿಜಯವು ತನ್ನನ್ನು ತಾನು ಗೆಲ್ಲುವುದರಲ್ಲಿ ಅಡಗಿದೆ. ಅದಕ್ಕಿಂತಲೂ ಮಿಗಿಲಾದ ವಿಜಯವು ಮತ್ತೊಂದಿಲ್ಲ. ಪ್ರವಾದಿಯಾದ ಮಹಮದ್ ಪೈಗಂಬರ್ ಅವರು “ಶ್ರದ್ಧೆಯೇ ನನ್ನ ಶಕ್ತಿಯ ಮೂಲ, ದುಃಖವೇ ನನ್ನ ಮಿತ್ರ, ಜ್ಞಾನವೇ ನನ್ನ ಆಯುಧ, ತಾಳ್ಮೆಯೇ ನನ್ನ ಕವಚ” ಎಂದಿದ್ದಾರೆ. ಧರ್ಮ ಸಂಸ್ಥಾಪನೆಯಂತಹ ಮಹತ್ತರ ಕಾರ್ಯ ಜರುಗಲು ಇವೆಲ್ಲವೂ ಅವರಿಗೆ ಪೂರಕವಾಗಿದ್ದವು. ವಿಕ್ಟರ್ ಹ್ಯೂಗೋರವರು  “ನಿದ್ರಿಸುವುದಕ್ಕಿಂತ ಮುಂಚೆ ಶುಭ ನಿರೀಕ್ಷೆ, ಪ್ರೀತಿ, ಕ್ಷಮೆ ಇವುಗಳನ್ನು ನಿಮ್ಮ ದಿಂಬುಗಳನ್ನಾಗಿಸಿಕೊಳ್ಳಿ. ಆಗ ಆನಂದದಿಂದ ಬೆಳಿಗ್ಗೆ ನೀವು ಏಳುವಿರಿ” ಎನ್ನುತ್ತಾರೆ. ನಾವು ಒಳ್ಳೆಯದನ್ನೇ ಚಿಂತಿಸುತ್ತಾ ಶುಭವನ್ನೇ ಬಯಸುತ್ತಾ ಪ್ರೀತಿಯುತವಾಗಿ ಬದುಕಿದರೆ ಆನಂದಮಯವಾದಂತಹ ಜೀವನವನ್ನು ನಮ್ಮದಾಗಿಸಿಕೊಳ್ಳಬಹುದು. ದೃಷ್ಟಿಯಂತೆ ಸೃಷ್ಟಿ ಎಂಬಂತೆ ವ್ಯಕ್ತಿಯ ದೃಷ್ಟಿಯು ಒಳ್ಳೆಯದಾಗಿದ್ದರೆ, ಆತನ ಆಲೋಚನೆಗಳೆಲ್ಲವೂ ಶ್ರೇಷ್ಠವಾಗಿರುತ್ತವೆ. ಆತನ ಚಿಂತನೆಗಳೇ ಕೆಟ್ಟತನದಿಂದ ಕೂಡಿದ್ದರೆ ಆತನನ್ನು ವಿನಾಶದ ಪ್ರಪಾತಕ್ಕೆ ತಳ್ಳುತ್ತವೆ. ಹಾಗಾಗಿ ನಾವು ಸುಚಿತ್ತವನ್ನು ಅರಳಿಸಿಕೊಳ್ಳಬೇಕು, ಸನಮಾರಗದಲ್ಲಿ ನಡೆಯಬೇಕು. ಇಂತಹ ನಡೆ - ನುಡಿಯು ಸರ್ವಾದರಣೀಯವಾಗಿರುತ್ತದೆ. ಬಸವಣ್ಣನವರು ಹೇಳಿರುವಂತೆ ಸಕಲರಿಗೂ ಲೇಸನ್ನು ಬಯಸುತ್ತಾ ಕಾಯಕವೇ ಕೈಲಾಸವೆಂಬ ಹಿತೋಕ್ತಿಯಂತೆ ಬದುಕು ಬಾಳು ಆನಂದಮಯವಾಗಿರುತ್ತದೆ. ಆಸೆಯೇ ದುಃಖಕ್ಕೆ ಮೂಲ ಕಾರಣ ಎಂಬುದನ್ನು ತಿಳಿದೂ ಮನುಷ್ಯ ಆಸೆಯ ಆಳಾಗುತ್ತಾನೆ. ಇಂತಹ ಆಸೆಗಳ ಈಡೇರಿಕೆಗಾಗಿ ಎಂತಹ ಕೆಟ್ಟ ಕಾರ್ಯಕ್ಕೂ ಕೈ ಹಾಕುತ್ತಾನೆ. ಸಮಾಜದಲ್ಲಿ ದುಷ್ಟವ್ಯಕ್ತಿಯಾಗಿ ಪರಿಗಣಿತನಾಗುತ್ತಾನೆ. ಆಸೆಯೇ ಜೀವಂತ ಬದುಕಿನ ಮೂಲ ಸೆಲೆ. ಆಸೆಯಿಲ್ಲದವನು ನಿರ್ಜೀವಿಯಂತೆ, ಆದರೆ ಆ ಆಸೆಯು ಇತಿಮಿತಿಯಲ್ಲಿರಬೇಕು. ಕವಿ ಬ್ರಹ್ಮಶಿವನು ತನ್ನ ಸಮಯ ಪರೀಕ್ಷೆ ಕೃತಿಯಲ್ಲಿ “ಅತಿಮೋಹಂ ಗತಿಗೆಡಿಸುಗುಂ” (ಅತಿಮೋಹವು ಗತಿಗೆಡಿಸುತ್ತದೆ) ಎಂದಿದ್ದಾನೆ. ಇಂತಹ ವ್ಯಕ್ತಿಗಳಿಗೆ ಈ ಸುಭಾಷಿತ ಮಾರ್ಗದರ್ಶಿಯಾಗಿದೆ.  “ಆಸೆಗಳಿಗೆ ಆಳಾದವನು ಲೋಕಕ್ಕೆ ಆಳು, ಆಸೆಯನ್ನು ಆಳುವವನಿಗೆ ಲೋಕವೇ ಆಳು” ಆಸೆಗಳಿಗೆ ತುತ್ತಾಗದೆ, ತನ್ನ ಮನಸ್ಸನ್ನು ಗಟ್ಟಿಗೊಳಿಸಿಕೊಂಡು ಬದುಕಿದರೆ ಆತನು ಸರ್ವಮಾನ್ಯನಾಗುತ್ತಾನೆ. ಬೇರೆಯವರಿಗೆ ತೊಂದರೆಯುಂಟುಮಾಡದೆ, ದುಷ್ಟರಿಗೆ ತಲೆಬಾಗದೆ, ಸತ್ಪುರುಷರು ತುಳಿದ ಮಾರ್ಗವನ್ನು ಬಿಡದೆ ಸ್ವಲ್ಪವೇ ಸಾಧಿಸಿದರೂ ದೊಡ್ಡದು, ಇಂತಹ ಜೀವನವೇ ಶ್ರೇಷ್ಟ. ಸತ್ಪುರುಷರನ್ನು ಮಾರ್ಗದರ್ಶಿಗಳನ್ನಾಗಿಸಿಕೊಂಡ ನಂತರ ಅವರ ಹಿರಿಮೆಯನ್ನು ಯಾವಾಗಲೂ ಅವರು ಅದನ್ನು ಸಾಧಿಸಲು ಬಳಸಿದ ಮಾರ್ಗದಿಂದ ಅರಿಯಬೇಕು. ಅನಂತರ ಅದನ್ನು ಅರ್ಥಮಾಡಿಕೊಂಡರೆ ಅದಕ್ಕಿಂತ ಶ್ರೇಷ್ಟವಾದ ಜೀವನ ಮತ್ತೊಂದಿಲ್ಲ. “ಉತ್ತಮರ ಸಂಪರ್ಕ ಯಾರಿಗೆ ತಾನೆ ಶ್ರೇಯಸ್ಸನ್ನು ಉಂಟುಮಾಡುವುದಿಲ್ಲ; ಕಮಲದ ಎಲೆ ಮೇಲೆ ಬಿದ್ದ ನೀರು ಮುತ್ತಿನ ಸೊಬಗನ್ನು ಪಡೆಯುವಂತೆ ಉತ್ತಮರ ಸಂಪರ್ಕ ಸನ್ಮಾರ್ಗದತ್ತ ಕೊಂಡೊಯ್ಯುತ್ತದೆ.”'
    ]

    speaker_wav_path_arr = [
        '/teamspace/studios/this_studio/TTS/kannada-dataset-subset/LDC-IL_Scheduled_Kannada_Female_21To50_Creative Text-T2_SP-0131_T2-0006.wav',
        '/teamspace/studios/this_studio/TTS/kannada-dataset-subset/LDC-IL_Scheduled_Kannada_Male_21To50_Person Name-W2_SP-0345_W2-0466.wav'
    ]

    speaker_gender = [
        'female',
        'male'
    ]
    text_index = 1
    speaker_index = 0
    file_index = 1
    wavs = synthesizer.tts(text_arr[text_index], speaker_wav=speaker_wav_path_arr[speaker_index],split_sentences=True,speaker_id = None)
    out = io.BytesIO()
    output_path = f'kn_synth_audio_op_synth_{text_to_mel_model_ckpt_offset}_sp_encoder_{speaker_encoder_ckpt_offset}_{speaker_gender[speaker_index]}_text_{text_index}_file_{file_index}.wav'
    synthesizer.save_wav(wavs, output_path)