# OpenBible-TTS

Welcome to OpenBible-TTS, a project to create a free, open-source, and high-quality text-to-speech (TTS) system from [OpenBible](https://open.bible/). This project is inspired by [masakhane-io/bibleTTS](https://github.com/masakhane-io/bibleTTS) and [coqui-ai/open-bible-scripts](https://github.com/coqui-ai/open-bible-scripts), with a twist of applying the more recent [Massively Multilingual Speech (MMS)](https://arxiv.org/abs/2305.13516) for audio alignment.

This project is only possible with the OpenBible's open-source friendly [CC BY-SA 4.0](https://open.bible/bibles/swahili-biblica-audio-bible/#by-sa) license. We are grateful for their work and dedication to making the Bible accessible to everyone.

## Languages

| Language | Dataset                                                                        | Text-to-Speech Model                                             |
| -------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Swahili  | [OpenBible Swahili](https://huggingface.co/datasets/bookbot/OpenBible_Swahili) | [VITS](https://huggingface.co/bookbot/vits-base-sw-KE-OpenBible) |

### Swahili (Kiswahili)

Currently, this project only supports the [Swahili: Biblica® Open Kiswahili Contemporary Version (Neno)](https://open.bible/bibles/swahili-biblica-audio-bible/) edition of OpenBible.

Initially, we wanted to use [coqui-ai/open-bible-scripts](https://github.com/coqui-ai/open-bible-scripts) to parse the USX/USFM files, but we found out that the [transcripts](https://open.bible/bibles/swahili-biblica-text-bible/) are in an unsupported format. We thereby developed our own [parser](./notebooks/scrape_openbible_audio.ipynb) to extract the verse texts using RegEx.

## Alignment

We followed the [CTC Forced Alignment API Tutorial](https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html) graciously provided by PyTorch. We used the [Massively Multilingual Speech (MMS)](https://arxiv.org/abs/2305.13516) model to align the audio to the text.

Like the challenges explained in the MMS paper, we also faced the same noisy audio alignment issues:

- Chapter introduction narration
- Occasional verse number reading
- Digits/number handling

While [masakhane-io/bibleTTS](https://github.com/masakhane-io/bibleTTS) proposes a solution by manually (1) inserting chapter introduction transcript and (2) spells out numbers, we decided to use a mix of MMS' method by (1) inserting `*` token for additional speech (e.g. chapter introduction, verse number reading) and (2) converting digits to words using [num2words](https://github.com/rhasspy/num2words), if available.

The aligned verse text and audio are then segmented into verse-level segments for TTS model training. We recommend reading the MMS paper for better understanding.

## Usage

### Installation

Firstly, install dependencies. It is important that you have `torchaudio>=2.1.0` installed, or any version that supports `torchaudio.pipelines.MMS_FA`.

```sh
git clone https://github.com/bookbot-hive/OpenBible-TTS.git
cd OpenBible-TTS
pip install -r requirements.txt
```

### Data Preparation

Secondly, prepare the dataset. OpenBible audio per-chapter files are usually in MP3 format. You first need to convert them to WAV format. We recommend using [ffmpeg](https://ffmpeg.org/). Our [segmentation script](./scripts/segment_audio.py) supports any arbitrary input audio sample rate. Our audio files are structured as follows

```
downloads/wavs_44/
├── 1CH
│   ├── 1CH_001.wav
│   ├── 1CH_002.wav
│   ├── ...
├── 1CO
│   ├── 1CO_001.wav
│   ├── 1CO_002.wav
│   ├── ...
├── ...
...
```

As for the text, you can use the [parser](./notebooks/scrape_openbible_audio.ipynb) we developed to extract the verse texts *if* the formats are uniform. The parser will output JSON files that looks like follows:

```json
[
  {
    "verseNumber": "HAB 1:1",
    "verseText": "Neno alilopokea nabii Habakuki."
  },
  {
    "verseNumber": "HAB 1:2",
    "verseText": "Ee BWANA, hata lini nitakuomba msaada, lakini wewe husikilizi? Au kukulilia, \u201cUdhalimu!\u201d Lakini hutaki kuokoa?"
  },
  {
    "verseNumber": "HAB 1:3",
    "verseText": "Kwa nini unanifanya nitazame dhuluma? Kwa nini unavumilia makosa? Uharibifu na udhalimu viko mbele yangu; kuna mabishano na mapambano kwa wingi."
  }
]
```

Examples of these JSON files can be found in the [data](./data/openbible_swahili/) directory.

### Forced-Alignment and Segmentation

Then to align and segment the chapter-level audio to verse-level audio, run the following script:

```sh
python scripts/segment_audio.py \
    --audio_path downloads/wavs_44/PSA/PSA_119.wav \ # path to the audio file
    --json_path data/openbible_swahili/PSA.json \ # path to the JSON file
    --output_dir outputs/openbible_swahili/ \ # output directory
    --chunk_size_s 15 # chunk size in seconds
```

which will generate

```
outputs/openbible_swahili/PSA/
├── PSA_001
│   ├── PSA_001_001.txt
│   ├── PSA_001_001.wav
│   ├── PSA_001_002.txt
│   ├── PSA_001_002.wav
│   ├── ...
├── PSA_002
│   ├── PSA_002_001.txt
│   ├── PSA_002_001.wav
│   ├── PSA_002_002.txt
│   ├── PSA_002_002.wav
├── ...
...
```

where each verse-level segment is aligned to the corresponding verse text, and input audio sample rate is also maintained.

We also provided a [runner script](./scripts/run_segmentation.py) that can be used to segment all the audio files in a directory, typically for each book in the Bible. You can run it like follows:

```sh
python scripts/run_segmentation.py \
    --json_path data/openbible_swahili/PSA.json \
    --audio_dir downloads/wavs_44/PSA/ \
    --output_dir outputs/openbible_swahili/ \
    --chunk_size_s 15
```

Finally, we also provided a [bash script](./run_segmentation.sh) to segment all downloaded books.

### Probability-based Alignment Score Filtering

As proposed in §3.1.5 of [MMS](https://arxiv.org/abs/2305.13516), we implemented a length-normalized probability difference filtering to remove noisy alignments based on the following equation:

$$\frac{1}{T} \left[\log P\left(Y^{\text {aligned}} \mid X\right)-\log P\left(Y^{\text {greedy}} \mid X\right)\right]$$

where $T$ is the length of the audio, $P\left(Y^{\text{aligned}} \mid X\right)$ is the probability of the forced-alignment path, and $P\left(Y^{\text{greedy}} \mid X\right)$ is the probability of the greedy sequence.

Like MMS, we select `−0.2` as the default threshold and choose samples with scores greater than this threshold.

The filtering script can be run as follows:

```sh
# score: -0.005685280751179646 (good alignment; accept)
python scripts/filter_audio.py \
    --audio_path outputs/openbible_swahili/EPH/EPH_003/EPH_003_001.wav \
    --ground_truth "kwa sababu hii mimi paulo mfungwa wa kristo yesu kwa ajili yenu ninyi watu wa mataifa" \
    --chunk_size_s 15

# score: -0.5496844846810868 (bad alignment; reject)
python scripts/filter_audio.py \
    --audio_path outputs/openbible_swahili/EPH/EPH_001/EPH_001_020.wav \
    --ground_truth "aliyoitumia katika kristo alipomfufua kutoka kwa wafu na akamketisha mkono wake wa kuume huko mbinguni" \
    --chunk_size_s 15
```

Likewise, we also provided a [runner script](./scripts/run_filter.py) that can be used to segment all the audio files in a directory, typically for each book in the Bible. You can run it like follows:

```sh
python scripts/run_filter.py \
    --audio_dir outputs/openbible_swahili/PSA/ \
    --output_dir outputs/openbible_swahili_filtered/ \
    --chunk_size_s 15 \
    --probability_difference_threshold -0.2
```

It will then generate a new directory with the filtered audio segments, retaining the same directory structure.

Finally, we also provided a [bash script](./run_filter.sh) to filter generated segments for all books.

## Future Improvements

- [ ] Support chunk batching
- [x] Probability-based alignment filtering
- [x] Support batched filtering
- [ ] CER-based filtering

## License

Our codes and scripts aree licensed under the [Apache License 2.0](./LICENSE). Contents of [data](./data/) are licensed under the [CC BY-SA 4.0](https://open.bible/bibles/swahili-biblica-audio-bible/#by-sa) license.

## References

```bibtex
@misc{pratap2023scaling,
    title={Scaling Speech Technology to 1,000+ Languages}, 
    author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
    year={2023},
    eprint={2305.13516},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

```bibtex
@misc{meyer2022bibletts,
    title={BibleTTS: a large, high-fidelity, multilingual, and uniquely African speech corpus}, 
    author={Josh Meyer and David Ifeoluwa Adelani and Edresson Casanova and Alp Öktem and Daniel Whitenack Julian Weber and Salomon Kabongo and Elizabeth Salesky and Iroro Orife and Colin Leong and Perez Ogayo and Chris Emezue and Jonathan Mukiibi and Salomey Osei and Apelete Agbolo and Victor Akinode and Bernard Opoku and Samuel Olanrewaju and Jesujoba Alabi and Shamsuddeen Muhammad},
    year={2022},
    eprint={2207.03546},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```