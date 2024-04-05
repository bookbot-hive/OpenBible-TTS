from pathlib import Path
import argparse

from tqdm.auto import tqdm

from segment_audio import segment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--json_path", required=True, help="Path to the JSON file. Example: data/openbible_swahili/PSA.json"
)
parser.add_argument(
    "--audio_dir",
    required=True,
    help="Path to the audio directory. Example: downloads/wavs_16/PSA/",
)
parser.add_argument("--output_dir", default="outputs/openbible_swahili/", help="Path to the output directory")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")


def main(args):
    audio_dir = Path(args.audio_dir)
    audios = sorted(audio_dir.rglob("*.wav"))
    for audio_path in tqdm(audios, desc=f"Segmenting {audio_dir.stem}"):
        segment(audio_path, args.json_path, args.output_dir, args.chunk_size_s)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
