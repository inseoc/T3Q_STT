from setproctitle import setproctitle
setproctitle("Wav2Vec2 Dataset")

from datasets import Dataset
from transformers import Wav2Vec2Processor
from tqdm import tqdm

import numpy as np
import os
import soundfile as sf
import pandas as pd
import time
import config.config as cfg

    
class DatasetMake():
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(cfg.path.processor_path)
        self.input_values_list = list()
        self.input_length_list = list()
        self.labels_list = list()
        
    def _read_txt_file(self, file_path: str) -> str: 
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return text
    
    def make_dataset_list(self):
        start = time.time()
        for folder in tqdm(os.listdir(kss_path), desc="total data reading"):                          # cpu mem이 부족하면 12분 정도, 널널하면 1분 정도 걸린다.
            folder_path = os.path.join(kss_path, folder)
            for file in tqdm(sorted(os.listdir(folder_path)), desc="sub data reading"):
                if file.endswith(".wav"):
                    wav_path = os.path.join(folder_path, file)
                    audio, _ = sf.read(wav_path)
                    if audio.ndim > 1:
                        audio = np.delete(audio, 1, axis=1)
                        audio = audio.reshape(-1)
                    input_value = self.processor(audio, sampling_rate=16000).input_values[0]
                    self.input_values_list.append(input_value)                  # processor를 거쳐 나온 오디으 array 값을 리스트에 저장
                    self.input_length_list.append(len(input_value))
                    txt_path = wav_path.replace(".wav", ".txt")
                    text = self._read_txt_file(txt_path)
                    with processor.as_target_processor():
                        self.labels_list.append(self.processor(text).input_ids)      # processor가 보유한 vocab에 맞는 인덱스 label값을 리스트에 저장
        print("total time:", time.time()-start)
        
        return self.input_values_list, self.input_length_list, self.labels_list
    
    def make_dataset_df(self):
        input_values_list, input_length_list, labels_list = self.make_dataset_list()
        train_idx = int(cfg.train.train_rate * len(input_values_list))

        train_df = pd.DataFrame({'input_values': input_values_list[:train_idx], 'input_length': input_length_list[:train_idx], 'labels': labels_list[:train_idx]})
        test_df = pd.DataFrame({'input_values': input_values_list[train_idx:], 'input_length': input_length_list[train_idx:], 'labels': labels_list[train_idx:]})
        
        return train_df, test_df
    
    def refine_trainset(self, train_timit):
        max_input_length_in_sec = cfg.max_input_length_in_sec
        train_timit = train_timit.filter(lambda x:x < max_input_length_in_sec * self.processor.feature_extractor.sampling_rate, input_columns=['input_length'])
        
        return train_timit
    
    def make_dataset(self):
        train_df, test_df = make_dataset_df()
        
        train_timit = Dataset.from_pandas(train_df)
        train_timit = self.refine_trainset(train_timit)
        test_timit = Dataset.from_pandas(test_df)
        
        print("trainset length:", len(train_timit))
        print("testset length:", len(test_timit))
        
        return train_timit, test_timit