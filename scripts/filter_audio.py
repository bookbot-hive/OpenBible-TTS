from pathlib import Path
import argparse
import json
import re
import string

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from unidecode import unidecode
from num2words import num2words
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_path",
    required=True,
    help="Path to the audio file. Example: outputs/openbible_swahili/EPH/EPH_003/EPH_003_001.wav",
)
parser.add_argument(
    "--json_path", required=True, help="Path to the JSON file. Example: data/openbible_swahili/EPH.json"
)
parser.add_argument("--output_dir", default="outputs/openbible_swahili/", help="Path to the output directory")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")

# MMS feature extractor minimum input frame size (25ms)
# also the same value as `ratio`
# `ratio = input_waveform.size(1) / num_frames`
SUBSAMPLING_RATIO = 400


def preprocess_verse(text: str) -> str:
    text = unidecode(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang="sw"), text)
    text = re.sub("\s+", " ", text)
    return text


def load_transcript(json_path: Path, verse: str) -> str:
    with open(json_path, "r") as f:
        data = json.load(f)

    # convert PSA 19:1 -> PSA_019_001
    get_verse = lambda x: x.split()[0] + "_" + x.split(":")[0].split()[1].zfill(3) + "_" + x.split(":")[1].zfill(3)
    # filter by verse
    transcript = [d["verseText"] for d in data if get_verse(d["verseNumber"]) == verse][0]
    return transcript


###############################################################################################################
# functions modified from https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
###############################################################################################################


def align(emission, tokens, device):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


def compute_alignment_scores(emission, transcript, dictionary, device):
    tokens = [dictionary[char] for word in transcript for char in word]
    _, scores = align(emission, tokens, device)
    return scores


def compute_probability_difference(audio_path: str, json_path: str, chunk_size_s: int = 15) -> float:
    audio_path = Path(audio_path)
    json_path = Path(json_path)

    # verse_id = "MAT_019_001"
    verse_id = audio_path.stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load MMS aligner model
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device)
    DICTIONARY = bundle.get_dict(star=None)

    # load transcript
    transcript = load_transcript(json_path, verse_id)
    # apply preprocessing
    verse = preprocess_verse(transcript)
    words = verse.split()

    # load audio
    input_waveform, input_sample_rate = torchaudio.load(audio_path)
    resampler = T.Resample(input_sample_rate, bundle.sample_rate, dtype=input_waveform.dtype)
    resampled_waveform = resampler(input_waveform)
    # split audio into chunks to avoid OOM and faster inference
    chunk_size_frames = chunk_size_s * bundle.sample_rate
    chunks = [
        resampled_waveform[:, i : i + chunk_size_frames]
        for i in range(0, resampled_waveform.shape[1], chunk_size_frames)
    ]

    # collect per-chunk emissions, rejoin
    emissions = []
    with torch.inference_mode():
        for chunk in chunks:
            # NOTE: we could pad here, but it'll need to be removed later
            # skipping for simplicity, since it's at most 25ms
            if chunk.size(1) >= SUBSAMPLING_RATIO:
                emission, _ = model(chunk.to(device))
                emissions.append(emission)

    emission = torch.cat(emissions, dim=1)
    num_frames = emission.size(1)
    assert len(DICTIONARY) == emission.shape[2]

    # method proposed in §3.1.5 of MMS paper: https://arxiv.org/abs/2305.13516
    # \frac{1}{T} \log P(Y_{aligned} | X) - \log P(Y_{greedy} | X)

    # compute greedy search score
    probs = torch.softmax(emission, dim=2)
    greedy_probs = torch.max(probs, dim=-1).values.squeeze()
    greedy_log_probs = torch.sum(torch.log(greedy_probs)).cpu().numpy().item()

    # compute forced-alignment score
    aligned_probs = compute_alignment_scores(emission, words, DICTIONARY)
    aligned_log_probs = torch.sum(torch.log(aligned_probs)).cpu().numpy().item()

    # compute length-normalized probability difference
    probability_diff = (aligned_log_probs - greedy_log_probs) / num_frames

    return probability_diff


if __name__ == "__main__":
    args = parser.parse_args()
    probability_diff = compute_probability_difference(
        args.audio_path, args.json_path, args.output_dir, args.chunk_size_s
    )