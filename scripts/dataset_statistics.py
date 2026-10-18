"""
Standalone dataset statistics script for our local MELD copy.
Does not import or modify anything from preprocess.py / dataloader.py / etc.
Reads the raw CSVs directly and recomputes the same style of table published
on https://affective-meld.github.io/, plus a per-class breakdown, plus the
"effective" 4-class subset that actually goes into MemoCMT training via
preprocess_MELD()'s meld2label filter.

Methodology note: the official site's stats-generation code isn't public, so
"# of unique words" and "# of emotion shift" below use the tokenization/shift
definitions that empirically landed closest to the published numbers (tested
against several alternatives on the train split):
  - unique words: lowercase + whitespace split, punctuation left attached to
    the token (plain `text.lower().split()`) -> 10,375 vs. official 10,643
    (closest of 6 variants tried; stripping punctuation gave only 5,456)
  - emotion shift: shift on Sentiment (not the 7-way Emotion), counted only
    between consecutive utterances with a different Speaker -> 3,806 vs.
    official 4,003 (closest of 4 variants tried)
Neither is an exact match (no access to the original tool), but both are
within ~5% instead of the ~30% gap the naive version had. This does not
affect the data that actually feeds MemoCMT: the per-class / per-split
counts below match the official table exactly regardless of this choice.
"""
import os
import json
from datetime import datetime

import pandas as pd

DATA_ROOT = r"C:\Users\USER\OneDrive\Desktop\local_server\CBALL_Kyle\MemoCMT_orig\data\MELD.Raw"
SPLITS = {
    "Train": "train_sent_emo.csv",
    "Dev": "dev_sent_emo.csv",
    "Test": "test_sent_emo.csv",
}

MELD2LABEL = {"anger": "ang", "joy": "hap", "neutral": "neu", "sadness": "sad"}


def parse_time(t):
    # format: hh:mm:ss,ms
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def tokenize(text):
    return str(text).lower().split()


def general_stats(df):
    all_words = set()
    utt_lengths = []
    for text in df["Utterance"]:
        words = tokenize(text)
        all_words.update(words)
        utt_lengths.append(len(words))

    emotions_per_dialogue = df.groupby("Dialogue_ID")["Emotion"].nunique()

    shifts = 0
    for _, g in df.sort_values("Utterance_ID").groupby("Dialogue_ID"):
        sentiments = g["Sentiment"].tolist()
        speakers = g["Speaker"].tolist()
        for i in range(1, len(sentiments)):
            if sentiments[i] != sentiments[i - 1] and speakers[i] != speakers[i - 1]:
                shifts += 1

    durations = []
    for st, et in zip(df["StartTime"], df["EndTime"]):
        try:
            durations.append(parse_time(et) - parse_time(st))
        except Exception:
            pass

    return {
        "num_modality": "{a,v,t}",
        "num_unique_words": len(all_words),
        "avg_utterance_length": round(sum(utt_lengths) / len(utt_lengths), 2),
        "max_utterance_length": max(utt_lengths),
        "avg_emotions_per_dialogue": round(emotions_per_dialogue.mean(), 2),
        "num_dialogues": df["Dialogue_ID"].nunique(),
        "num_utterances": len(df),
        "num_speakers": df["Speaker"].nunique(),
        "num_emotion_shift": shifts,
        "avg_duration_s": round(sum(durations) / len(durations), 2) if durations else None,
    }


def per_class_counts(df):
    return df["Emotion"].value_counts().to_dict()


def effective_4class_counts(df):
    mapped = df["Emotion"].map(MELD2LABEL)
    return mapped.value_counts().to_dict()


PIPELINE_VERSION = {
    "preprocess_source": "scripts/preprocess.py :: preprocess_MELD()",
    "label_filter": "4-class: anger/joy/neutral/sadness -> ang/hap/neu/sad (disgust/fear/surprise dropped)",
    "fixes_applied": [
        "LABEL_MAP scoping bug (NameError on every sample without --all_classes)",
        "wav extraction now skips re-encoding if the file already exists",
    ],
    "note": (
        "This file describes the dataset only (which utterances/labels exist, "
        "pre audio-channel-handling). It is shared across every demo/run since "
        "preprocess.py's output (train/val/test.pkl) does not depend on the "
        "audio_channel_mode training config -- that choice (mean/left/right "
        "downmix) is applied live in dataloader.py and only affects individual "
        "runs, documented separately in each results/<demo_name>/ folder."
    ),
}


def main():
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_root": DATA_ROOT,
        "pipeline_version": PIPELINE_VERSION,
        "splits": {},
    }

    print(f"{'Statistics':32}{'Train':>10}{'Dev':>10}{'Test':>10}")
    general_by_split = {}
    for split_name, csv_name in SPLITS.items():
        df = pd.read_csv(os.path.join(DATA_ROOT, csv_name))
        general_by_split[split_name] = general_stats(df)
        report["splits"][split_name] = {
            "general": general_by_split[split_name],
            "per_class_7way": per_class_counts(df),
            "per_class_4way_effective": effective_4class_counts(df),
        }

    keys = [
        ("num_modality", "# of modality"),
        ("num_unique_words", "# of unique words"),
        ("avg_utterance_length", "Avg. utterance length"),
        ("max_utterance_length", "Max. utterance length"),
        ("avg_emotions_per_dialogue", "Avg. # of emotions per dialogue"),
        ("num_dialogues", "# of dialogues"),
        ("num_utterances", "# of utterances"),
        ("num_speakers", "# of speakers"),
        ("num_emotion_shift", "# of emotion shift"),
        ("avg_duration_s", "Avg. duration of an utterance (s)"),
    ]
    for key, label in keys:
        row = f"{label:32}"
        for split_name in SPLITS:
            row += f"{str(general_by_split[split_name][key]):>10}"
        print(row)

    print("\nPer-class counts (all 7 original MELD emotions):")
    print(f"{'Emotion':12}{'Train':>8}{'Dev':>8}{'Test':>8}")
    all_classes = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    for cls in all_classes:
        row = f"{cls.capitalize():12}"
        for split_name in SPLITS:
            row += f"{report['splits'][split_name]['per_class_7way'].get(cls, 0):>8}"
        print(row)

    print("\nEffective 4-class subset actually used by MemoCMT (ang/hap/sad/neu):")
    print(f"{'Class':8}{'Train':>8}{'Dev':>8}{'Test':>8}")
    for cls in ["ang", "hap", "sad", "neu"]:
        row = f"{cls:8}"
        for split_name in SPLITS:
            row += f"{report['splits'][split_name]['per_class_4way_effective'].get(cls, 0):>8}"
        print(row)

    out_dir = os.path.join(os.path.dirname(__file__), "results", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "dataset_statistics.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {out_json}")
    print("(shared across all demos -- not duplicated per run_name folder)")


if __name__ == "__main__":
    main()
