from typing import List, Tuple
from pathlib import Path
import argparse
import json
import re
import string

import torch
import torchaudio
import torchaudio.functional as F

from unidecode import unidecode
from num2words import num2words
from scipy.io.wavfile import write
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument(
    "--json_path", required=True, help="Path to the JSON file. Example: data/openbible_swahili/PSA.json"
)
parser.add_argument(
    "--audio_path",
    required=True,
    help="Path to the audio file, must be 16kHz. Example: downloads/wavs_16/PSA/PSA_119.wav",
)
parser.add_argument("--output_dir", default="outputs/openbible_swahili/", help="Path to the output directory")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")


def preprocess_verse(text: str) -> str:
    text = unidecode(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang="sw"), text)
    text = re.sub("\s+", " ", text)
    return text


def load_transcripts(json_path: Path, chapter: str) -> Tuple[List[str], List[str]]:
    with open(json_path, "r") as f:
        data = json.load(f)

    # convert PSA 19:1 -> PSA_019
    get_chapter = lambda x: x.split()[0] + "_" + x.split(":")[0].split()[1].zfill(3)
    # filter by book and chapter
    transcripts = [d["verseText"] for d in data if get_chapter(d["verseNumber"]) == chapter]
    verse_ids = [d["verseNumber"] for d in data if get_chapter(d["verseNumber"]) == chapter]
    return verse_ids, transcripts


###############################################################################################################
# functions taken from https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
###############################################################################################################


def align(emission, tokens, device):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def compute_alignments(emission, transcript, dictionary, device):
    tokens = [dictionary[char] for word in transcript for char in word]
    alignment, scores = align(emission, tokens, device)
    token_spans = F.merge_tokens(alignment, scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    return word_spans


def main(args):
    audio_path = Path(args.audio_path)
    json_path = Path(args.json_path)

    # book = "MAT"; chapter = "MAT_019"
    book, chapter = json_path.stem, audio_path.stem

    # prepare output directories
    output_dir = Path(args.output_dir) / book / chapter
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load MMS aligner model
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=True).to(device)
    DICTIONARY = bundle.get_dict()

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
    waveform, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Sample rate must be 16kHz!"
    # split audio into chunks to avoid OOM and faster inference
    chunk_size_frames = args.chunk_size_s * sr
    chunks = [waveform[:, i : i + chunk_size_frames] for i in range(0, waveform.shape[1], chunk_size_frames)]

    # collect per-chunk emissions, rejoin
    emissions = []
    with torch.inference_mode():
        for chunk in chunks:
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
        ratio = waveform.size(1) / num_frames
        x0 = int(ratio * verse_spans[0][0].start)
        x1 = int(ratio * verse_spans[-1][-1].end)
        transcript = " ".join(verse_words)
        segment = waveform[:, x0:x1]
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
        write(audio_path, bundle.sample_rate, segment.squeeze().numpy())

        # write transcript
        transcript_path = (output_dir / verse_file_name).with_suffix(".txt")
        with open(transcript_path, "w") as f:
            f.write(label)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
