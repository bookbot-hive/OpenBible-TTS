from pathlib import Path
import argparse
import shutil

from tqdm.auto import tqdm

from filter_audio import compute_probability_difference, compute_probability_difference_batched

parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_dir",
    required=True,
    help="Path to the audio directory, must contain txt files with the same name. Example: outputs/openbible_swahili/PSA/",
)
parser.add_argument("--output_dir", default="outputs/openbible_swahili_filtered/", help="Path to the output directory")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")
parser.add_argument(
    "--probability_difference_threshold",
    type=float,
    default=-0.2,
    help="Probability difference threshold for filtering. Default: -0.2 from MMS.",
)
parser.add_argument(
    "--batched",
    action="store_true",
    help="Whether to batch-filter. Currently still just on-par with non-batched filtering.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Batch size for batch-filtering. Default to 16 (usable for P100 16GB).",
)


def main(args):
    audio_dir = Path(args.audio_dir)
    audios = sorted(audio_dir.rglob("*/*.wav"))
    if not args.batched:
        for audio_path in tqdm(audios, desc=f"Filtering {audio_dir.stem}"):
            transcript_path = audio_path.with_suffix(".txt")
            # create output directory `output_dir/{book}/{chapter}/`
            output_path = Path(args.output_dir) / audio_dir.stem / audio_path.parent.stem
            output_audio_path = output_path / audio_path.name
            output_transcript_path = output_path / transcript_path.name
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)

            # read ground truth
            with open(transcript_path) as f:
                ground_truth = f.read()

            # compute probability difference
            probability_difference = compute_probability_difference(audio_path, ground_truth, args.chunk_size_s)

            # copy audio and transcript if probability_difference is greater than threshold
            if probability_difference > args.probability_difference_threshold:
                shutil.copy(audio_path, output_audio_path)
                shutil.copy(transcript_path, output_transcript_path)
    else:
        # read ground truth
        transcript_paths = [audio.with_suffix(".txt") for audio in audios]
        ground_truths = []
        for transcript_path in transcript_paths:
            with open(transcript_path) as f:
                ground_truths.append(f.read())

        probability_differences = compute_probability_difference_batched(audios, ground_truths, args.batch_size)
        for audio_path, probability_difference in zip(audios, probability_differences):
            transcript_path = audio_path.with_suffix(".txt")
            # create output directory `output_dir/{book}/{chapter}/`
            output_path = Path(args.output_dir) / audio_dir.stem / audio_path.parent.stem
            output_audio_path = output_path / audio_path.name
            output_transcript_path = output_path / transcript_path.name
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)

            # copy audio and transcript if probability_difference is greater than threshold
            if probability_difference > args.probability_difference_threshold:
                shutil.copy(audio_path, output_audio_path)
                shutil.copy(transcript_path, output_transcript_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
