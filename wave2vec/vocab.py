from kspon_preprocess import special_filter, bracket_filter
from datasets import Dataset
from tqdm import tqdm
from glob import glob

import os
import re
import json
import random
import librosa


class Vocab():
    def __init__(self, args, cfg, pm):
        self.args = args
        self.cfg = cfg
        self.pm = pm
        self.text_list = list()
        self.audio_list = list()
    
    def remove_special_characters(self, batch):
        batch["text"] = special_filter(bracket_filter(batch["text"]))

        return batch

    def extract_all_chars(self, batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        
        return {"vocab": [vocab], "all_text": [all_text]}

    def vocab_data_make(self):
        print("Vocab Dataset Make Start...")

        ## ver1. kss
        if self.args.train_ver == "kss":
            kss_texts = sorted(glob(os.path.join(self.cfg.kss_path, '**', '*.txt'), recursive=True))
            for text_file in tqdm(kss_texts):
                with open(text_file, 'r') as f:
                    text_data = f.read()
            self.text_list.append(text_data)

        ## ver2. zeroth
        elif self.args.train_ver == "zeroth":
            zeorth_texts = sorted(glob(os.path.join(self.cfg.train_zeroth_path, '**', '*.txt'), recursive=True))
            for text_file in tqdm(zeorth_texts):
                with open(text_file, 'r') as f:
                    text_data = f.read()
            self.text_list.append(text_data)

        ## ver4. ksponspeech
        elif self.args.train_ver == "kspon":
            durations = 0
            max_sec = self.pm.max_sec
            min_sec = self.pm.min_sec

            kspon_wavs = sorted(glob(os.path.join(self.cfg.kspon_path, '**', '*.wav'), recursive=True))
            random.seed(44)
            random.shuffle(kspon_wavs)

            remove_re = '[a-zA-Z0-9%]'

            for file in tqdm(kspon_wavs):
                duration = librosa.get_duration(filename=file, sr=16000)
                if min_sec < duration < max_sec:
                    text_path = file.replace(".wav", ".txt")
                    with open(text_path, 'r', encoding='cp949') as f:
                        text_data = f.read()
                    text = special_filter(bracket_filter(text_data))
                    if re.findall(remove_re, text) == []:
                        self.text_list.append(text)
                        self.audio_list.append(file)
                        durations += duration
                if durations >= self.pm.time_limit:
                    break

        print("vocab dataset length: ", len(self.text_list))

        return self.text_list, self.audio_list

    def vocab_dict_make(self, text_list):
        text_dict = {"text": text_list}

        vocab_timit = Dataset.from_dict(text_dict)
        vocab_timit = vocab_timit.map(self.remove_special_characters)
        vocabs = self.extract_all_chars(vocab_timit)
        vocab_list = list(set(vocabs["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        print("vocab length: ", len(vocab_dict))
        
        self.vocab_dict = vocab_dict
        
        return self.vocab_dict

    def vocab_file_make(self):
        text_list, audio_list = self.vocab_data_make()
        vocab_dict = self.vocab_dict_make(text_list)
        vocab_path = os.path.join(self.cfg.dataset_path, 'vocab.json')
        with open(vocab_path, 'w') as vocab_file:
            json.dump(self.vocab_dict, vocab_file)

        print("Vocab json file Make!")

        return vocab_path, text_list, audio_list