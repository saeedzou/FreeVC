# This script performs data augmentation by using FreeVC to convert a given source audio to a target audio.
# To guarantee diverse conversion outputs, the source speaker information is compared with multiple target speakers.
# And the target speakers with the highest cosine distance are selected.
# The selected target speakers is then used to convert the source audio.
# The converted audio is saved to the output directory.
#
# Required arguments:
#  --hpfile: The hyperparameter file path.
#  --ptfile: The checkpoint file path.
#  --spkmodel: The speaker embedding model name.
import os
import argparse
import numpy as np
import torch
import random
import librosa
from scipy.io.wavfile import write
from typing import Dict, List, Tuple
from tqdm import tqdm
from functools import lru_cache

import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from speaker_encoder.voice_encoder import SpeakerEncoder
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.spatial.distance import cdist




class FreeVC:
    def __init__(self, hpfile: str, ptfile: str):
        self.hps = utils.get_hparams_from_file(hpfile)
        self.device = utils.get_device()

        print("Loading model...")
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).to(self.device)
        _ = self.net_g.eval()
        print("Loading checkpoint...")
        _ = utils.load_checkpoint(ptfile, self.net_g, None, True)

        print("Loading WavLM for content...")
        self.cmodel = utils.get_cmodel(0)

        if self.hps.model.use_spk:
            print("Loading speaker encoder...")
            self.smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt', device=self.device)
    
    @torch.no_grad
    def convert(self, src_path: str, tgt_path: str) -> np.ndarray:
        wav_tgt, _ = librosa.load(tgt_path, sr=self.hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        if self.hps.model.use_spk:
            g_tgt = torch.from_numpy(self.smodel.embed_utterance(wav_tgt)).to(self.device).unsqueeze(0)
        else:
            wav_tgt = torch.from_numpy(wav_tgt).to(self.device).unsqueeze(0)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                self.hps.data.filter_length,
                self.hps.data.n_mel_channels,
                self.hps.data.sampling_rate,
                self.hps.data.hop_length,
                self.hps.data.win_length,
                self.hps.data.mel_fmin,
                self.hps.data.mel_fmax
            )

        # src
        wav_src = torch.from_numpy(librosa.load(src_path, sr=self.hps.data.sampling_rate)[0]).to(self.device).unsqueeze(0)
        c = utils.get_content(self.cmodel, wav_src)

        if self.hps.model.use_spk:
            audio = self.net_g.infer(c, g=g_tgt)
        else:
            audio = self.net_g.infer(c, mel=mel_tgt)
        return audio[0][0].data.cpu().float().numpy()

@lru_cache(maxsize=None)
def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    return librosa.load(file_path, sr=sr)[0]

def calculate_speaker_embedding(speaker_path: str, speaker_model: PretrainedSpeakerEmbedding, 
                                device: str = "cuda") -> np.ndarray:
    # The speaker path contains the path to the speaker audio files.
    embeddings = []
    for audio_file in os.listdir(speaker_path):
        if audio_file.endswith(".wav"):
            wav = load_audio(os.path.join(speaker_path, audio_file), sr=16000)
            # audio should be of shape (batch_size, n_channels, n_samples)
            wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device)
            embedding = speaker_model(wav)
            embeddings.append(embedding)
    # average the embeddings
    embeddings = np.mean(embeddings, axis=0)
    return embeddings

def calculate_speaker_embeddings(speakers_path: str, speaker_model: PretrainedSpeakerEmbedding, 
                                 device: str = "cuda") -> Dict[str, np.ndarray]:
    return {speaker: calculate_speaker_embedding(os.path.join(speakers_path, speaker), speaker_model, device)
            for speaker in os.listdir(speakers_path)}

def distance(embedding1, embedding2):
    return cdist(embedding1, embedding2, metric="cosine")

def select_target_speakers(src_embedding : np.ndarray, 
                           target_embeddings : Dict[str, np.ndarray], n : int = 3) -> List[Tuple[str, float]]:
    distances = {target: distance(src_embedding, target_embedding) \
                 for target, target_embedding in target_embeddings.items()}
    return sorted(distances.items(), key=lambda x: x[1], reverse=True)[:n]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)
    free_vc = FreeVC(args.hpfile, args.ptfile)
    speaker_model = PretrainedSpeakerEmbedding(args.spkmodel, device=device)
    
    target_embeddings = calculate_speaker_embeddings(args.target_spk_path, speaker_model, device)
    src_embedding = calculate_speaker_embedding(args.source_spk_path, speaker_model, device)
    target_speakers = select_target_speakers(src_embedding, target_embeddings)
    
    for i, (target, _) in enumerate(target_speakers):
        target_path = os.path.join(args.target_spk_path, target)
        target_file = random.choice([f for f in os.listdir(target_path) if f.endswith('.wav')])
        target = os.path.join(target_path, target_file)
        audio = free_vc.convert(args.source_wav_path, target)
        write(f"{args.save_path}/{os.path.basename(args.source_spk_path).split('.wav')[0]}_{i}", 16000, audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", required=True, help="The hyperparameter file path.", default='configs/freevc.json')
    parser.add_argument("--ptfile", required=True, help="The checkpoint file path.", default='checkpoints/freevc.pth')
    parser.add_argument("--spkmodel", required=True, help="The speaker embedding model name.", default='speechbrain/spkrec-ecapa-voxceleb')
    parser.add_argument("--target_spk_path", required=True, help="The path to the target speakers.")
    parser.add_argument("--source_spk_path", required=True, help="The path to the source speaker.")
    parser.add_argument("--source_wav_path", required=True, help="The path to the source wav file.")
    parser.add_argument("--save_path", required=True, help="The path to save the converted audio.", default='output')
    args = parser.parse_args()
    main(args)