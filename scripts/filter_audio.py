from pathlib import Path
import argparse

import torch
import torchaudio
import torchaudio.transforms as T

from utils import MMS_SUBSAMPLING_RATIO, preprocess_verse, compute_alignment_scores

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_path",
    required=True,
    help="Path to the audio file. Example: outputs/openbible_swahili/EPH/EPH_003/EPH_003_001.wav",
)
parser.add_argument("--ground_truth", required=True, help="Ground truth text to forced-align with.")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")


def compute_probability_difference(audio_path: str, ground_truth: str, chunk_size_s: int = 15) -> float:
    audio_path = Path(audio_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load MMS aligner model
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device)
    DICTIONARY = bundle.get_dict(star=None)

    # apply preprocessing
    verse = preprocess_verse(ground_truth)
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
            if chunk.size(1) >= MMS_SUBSAMPLING_RATIO:
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
    aligned_probs = compute_alignment_scores(emission, words, DICTIONARY, device)
    aligned_log_probs = torch.sum(torch.log(aligned_probs)).cpu().numpy().item()

    # compute length-normalized probability difference
    probability_diff = (aligned_log_probs - greedy_log_probs) / num_frames

    return probability_diff


if __name__ == "__main__":
    args = parser.parse_args()
    probability_difference = compute_probability_difference(args.audio_path, args.ground_truth, args.chunk_size_s)
    print(probability_difference)
