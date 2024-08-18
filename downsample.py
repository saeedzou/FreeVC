import os
import argparse
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

def process(wav_name, speaker):
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path):
        os.makedirs(os.path.join(args.out_dir1, speaker), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
        
        # Load the audio file
        wav, sr = librosa.load(wav_path, sr=48000)
        
        # Trim silence from the audio
        wav, _ = librosa.effects.trim(wav, top_db=20)
        
        # Normalize the audio if necessary
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        
        # Resample the audio to the two target sample rates
        wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        
        # Define the save paths for the resampled files
        save_path1 = os.path.join(args.out_dir1, speaker, wav_name)
        save_path2 = os.path.join(args.out_dir2, speaker, wav_name)
        
        # Save the resampled audio using soundfile
        sf.write(save_path1, wav1, args.sr1, subtype='PCM_16')
        sf.write(save_path2, wav2, args.sr2, subtype='PCM_16')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--sr2", type=int, default=22050, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="/home/Datasets/lijingyi/data/vctk/wav48_silence_trimmed/", help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default="./dataset/vctk-16k", help="path to target dir")
    parser.add_argument("--out_dir2", type=str, default="./dataset/vctk-22k", help="path to target dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir1, exist_ok=True)
    os.makedirs(args.out_dir2, exist_ok=True)

    # Process each file sequentially
    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            wav_files = os.listdir(spk_dir)
            for wav_name in tqdm(wav_files, desc=f'Processing {spk_dir}'):
                process(wav_name, speaker)
