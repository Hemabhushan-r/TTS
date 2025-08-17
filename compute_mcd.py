from pymcd.mcd import Calculate_MCD

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics 
mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")

# two inputs w.r.t. reference (ground-truth) and synthesized speeches, respectively

gt_audio_path = '/teamspace/studios/this_studio/TTS/kannada-dataset-subset/LDC-IL_Scheduled_Kannada_Female_21To50_Creative Text-T2_SP-0131_T2-0006.wav'
synth_audio_path = '/teamspace/studios/this_studio/TTS/kn_synth_audio_op_synth_3002_sp_encoder_1468_female_text_1_file_1.wav'
mcd_value = mcd_toolbox.calculate_mcd(gt_audio_path,synth_audio_path)


print(f'MCD Value : {mcd_value}')