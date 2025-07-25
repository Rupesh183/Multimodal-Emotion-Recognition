import os
import re
import pandas as pd


class IemocapPreprocessor:
    def __init__(self, dataset_path: str, sessions_count: int = 5):
        self._dataset_path = dataset_path
        self._sessions_count = sessions_count
        self._info_line = re.compile(r"\[.+\]\n", re.IGNORECASE)

    def generate_dataframe(self) -> pd.DataFrame:
        audios = []
        emotions = []
        texts = []

        for session in range(1, self._sessions_count + 1):
            session_dir = f"Session{session}"
            emo_eval_dir = os.path.join(
                self._dataset_path, session_dir, "dialog", "EmoEvaluation"
            )
            transcriptions_dir = os.path.join(
                self._dataset_path, session_dir, "dialog", "transcriptions"
            )
            if not os.path.exists(emo_eval_dir) or not os.path.exists(transcriptions_dir):
                continue

            eval_files = [file for file in os.listdir(emo_eval_dir) if "Ses" in file]
            for file in eval_files:
                eval_path = os.path.join(emo_eval_dir, file)
                transcription_path = os.path.join(transcriptions_dir, file)
                if not os.path.exists(transcription_path):
                    continue

                with open(eval_path, "r", encoding="utf-8", errors="ignore") as eval_file, open(
                    transcription_path, "r", encoding="utf-8", errors="ignore"
                ) as text_file:
                    eval_content = eval_file.read()
                    info_lines = re.findall(self._info_line, eval_content)
                    text_lines = sorted(
                        [
                            line
                            for line in text_file.readlines()
                            if line.startswith("Ses")
                            and line.split(" ")[0][-3:].isdigit()
                        ]
                    )
                    for info_line, text_line in zip(info_lines[1:], text_lines):
                        parts = info_line.strip().split("\t")
                        if len(parts) < 3:
                            continue
                        wav_file_name, emotion = parts[1:3]
                        if emotion == "xxx":
                            continue
                        if "]: " not in text_line:
                            continue
                        left_part, text = text_line.strip().split("]: ", 1)
                        if wav_file_name != left_part.split(" ")[0]:
                            continue
                        audios.append(wav_file_name)
                        emotions.append(emotion)
                        texts.append(text)

        return pd.DataFrame(data={"audio": audios, "text": texts, "emotion": emotions})
