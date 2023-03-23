import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from functools import partial

# Prepare TAT dataset in such format (https://github.com/voidful/asr-training):
"""
path,text
/xxx/2.wav,被你拒絕而記仇
/xxx/4.wav,電影界的人
/xxx/7.wav,其實我最近在想
"""

CLEAN_MAPPING = {
    "﹖": "?",
    "！": "!",
    "％": "%",
    "（": "(",
    "）": ")",
    "，": ",",
    "：": ":",
    "；": ";",
    "？": "?",
    "—": "--",
    "─": "-",
}
ACCEPTABLE_CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "àáéìîòúāō"
    " "
    '!"%()+,-./:;=?_~'
    "‘’“”'"
    "…⋯"
    "、。『』"
    "－"
)


def get_wav_from_txt(txt_path, wav_dir):
    f_name = txt_path.stem
    spk_name = txt_path.parent.name
    pattern = f"{f_name}-[0-9]*.wav"
    [wav_path] = (wav_dir / spk_name).glob(pattern)
    return wav_path


def get_transcription_from_json(json_path, transcript_type):
    with open(json_path) as f:
        json_data = json.load(f)
        txt = json_data[transcript_type]
        txt = clean_text(txt, transcript_type)
        return txt


def clean_text(txt, transcript_type):
    if transcript_type == "台羅數字調":
        txt = txt.strip()
        txt = txt.replace("'", " ")
        txt = txt.replace('"', " ")
        txt = txt.replace("“", " ")
        txt = txt.replace("”", " ")
        txt = txt.replace(":", " ")
        txt = txt.replace(")", " ")
        txt = txt.replace("(", " ")
        txt = txt.strip()
        if txt.endswith(","):
            txt = txt[:-1] + "."
        if txt[-1] not in "?!.":
            txt += "."
        return txt
    else:
        raise NotImplementedError


def validate_transcription(transcript, transcript_type, bad_count, verbose_fp=None):
    cleaned_transcript = transcript
    if transcript_type == "台羅數字調":
        for bad_char, good_char in CLEAN_MAPPING.items():
            cleaned_transcript = cleaned_transcript.replace(bad_char, good_char)
        for c in cleaned_transcript:
            if c not in ACCEPTABLE_CHARS:
                print(f"{bad_count + 1}\t{c}\t: {cleaned_transcript}", file=verbose_fp)
                return None, bad_count + 1
        return cleaned_transcript, bad_count
    else:
        raise NotImplementedError
        return cleaned_transcript, -1


def main():
    ############
    #  Config  #
    ############
    transcript_type = "台羅數字調"  # 台羅 or 漢羅台文 or 台羅數字調
    wav_type = "condenser"

    TAT_root = "/storage/speech_dataset/TAT/TAT-Vol1-train"
    output_path = "./TAT-Vol1-train.csv"

    TAT_root = Path(TAT_root).resolve()
    TAT_txt_dir = TAT_root / "json"
    TAT_wav_dir = TAT_root / wav_type / "wav"

    TAT_txt_list = list(TAT_txt_dir.rglob("*.json"))
    TAT_wav_list = [get_wav_from_txt(txt_path, TAT_wav_dir) for txt_path in tqdm(TAT_txt_list)]

    assert len(TAT_txt_list) == len(TAT_wav_list)

    tqdm.pandas()
    TAT_df = pd.DataFrame(
        {
            "json_path": TAT_txt_list,
            "wav_path": TAT_wav_list,
        }
    )
    get_transcript = partial(get_transcription_from_json, transcript_type=transcript_type)
    TAT_df["transcription"] = TAT_df["json_path"].map(get_transcript)

    bad_count = 0
    output_buffer = []
    for idx, data in TAT_df[["wav_path", "transcription"]].iterrows():
        wav_path = data["wav_path"]
        transcript = data["transcription"]
        result, bad_count = validate_transcription(transcript, transcript_type, bad_count)
        if result is not None:
            output_buffer.append(
                [
                    wav_path,
                    result,
                ]
            )

    pd.DataFrame(output_buffer).to_csv(output_path, index=None, header=["path", "text"])
    print("Output at", output_path)


if __name__ == "__main__":
    main()
