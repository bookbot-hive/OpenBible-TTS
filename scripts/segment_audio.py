from typing import List, Tuple
from pathlib import Path
import argparse
import json

import torch
import torchaudio
import torchaudio.transforms as T

from scipy.io.wavfile import write

from utils import MMS_SUBSAMPLING_RATIO, preprocess_verse, compute_alignments


parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_path",
    required=True,
    help="Path to the audio file. Example: downloads/wavs_44/PSA/PSA_119.wav",
)
parser.add_argument(
    "--json_path", required=True, help="Path to the JSON file. Example: data/openbible_swahili/PSA.json"
)
parser.add_argument("--output_dir", default="outputs/openbible_swahili/", help="Path to the output directory")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load MMS aligner model
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=True).to(device)
DICTIONARY = bundle.get_dict()


def load_transcripts(json_path: Path, chapter: str) -> Tuple[List[str], List[str]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    # convert PSA 19:1 -> PSA_019
    get_chapter = lambda x: x.split()[0] + "_" + x.split(":")[0].split()[1].zfill(3)
    # filter by book and chapter
    transcripts = [d["verseText"] for d in data if get_chapter(d["verseNumber"]) == chapter]
    verse_ids = [d["verseNumber"] for d in data if get_chapter(d["verseNumber"]) == chapter]
    return verse_ids, transcripts


def segment(audio_path: str, json_path: str, output_dir: str, chunk_size_s: int = 15):
    audio_path = Path(audio_path)
    json_path = Path(json_path)

    # book = "MAT"; chapter = "MAT_019"
    book, chapter = json_path.stem, audio_path.stem

    # prepare output directories
    output_dir = Path(output_dir) / book / chapter
    output_dir.mkdir(parents=True, exist_ok=True)

    # skip if already segmented
    if any(output_dir.iterdir()):
        print(f"Skipping {chapter}")
        return

    # load transcripts
    verse_ids, transcripts = load_transcripts(json_path, chapter)
    # apply preprocessing
    verses = [preprocess_verse(v) for v in transcripts]

    # insert "*" before every verse for chapter intro or verse number
    # see MMS robust noisy audio alignment
    # https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    augmented_verses = ["*"] * len(verses) * 2
    augmented_verses[1::2] = verses

    words = [verse.split() for verse in verses]
    augmented_words = [word for verse in augmented_verses for word in verse.split()]

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

    # perform forced-alignment
    word_spans = compute_alignments(emission, augmented_words, DICTIONARY, device)

    # remove "*" from alignment
    word_only_spans = [spans for spans, word in zip(word_spans, augmented_words) if word != "*"]
    assert len(word_only_spans) == sum(len(word) for word in words)

    # collect verse-level segments
    segments, labels, start = [], [], 0
    for verse_words in words:
        end = start + len(verse_words)
        verse_spans = word_only_spans[start:end]
        ratio = input_waveform.size(1) / num_frames
        x0 = int(ratio * verse_spans[0][0].start)
        x1 = int(ratio * verse_spans[-1][-1].end)
        transcript = " ".join(verse_words)
        segment = input_waveform[:, x0:x1]
        start = end
        segments.append(segment)
        labels.append(transcript)

    assert len(segments) == len(verse_ids) == len(labels)

    # export segments and forced-aligned transcripts
    for verse_id, segment, label in zip(verse_ids, segments, labels):
        # PSA 19:1 -> PSA_019_001
        verse_number = verse_id.split(":")[-1].zfill(3)
        verse_file_name = chapter + "_" + verse_number

        # write audio
        audio_path = (output_dir / verse_file_name).with_suffix(".wav")
        write(audio_path, input_sample_rate, segment.squeeze().numpy())

        # write transcript
        transcript_path = (output_dir / verse_file_name).with_suffix(".txt")
        with open(transcript_path, "w") as f:
            f.write(label)


if __name__ == "__main__":
    args = parser.parse_args()
    segment(args.audio_path, args.json_path, args.output_dir, args.chunk_size_s)
