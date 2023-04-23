import os
import torch
import re
import librosa
import wandb
import numpy as np
import pandas as pd
import soundfile as sf


from tqdm import tqdm
from datasets import Dataset, load_metric
from glob import glob
from kspon_preprocess import special_filter, bracket_filter, del_noise
from vocab import Vocab
from transformers import (Wav2Vec2CTCTokenizer, 
                         Wav2Vec2FeatureExtractor, 
                         Wav2Vec2Processor, 
                         Wav2Vec2ForCTC,
                         TrainingArguments,
                         Trainer)
                
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


def processor(vocab_path):
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    print("Processor Make!")
    
    return processor


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}
    
    
class TrainDataset():
    def __init__(self, args, pm, processor, text_list, audio_list=None):
        self.args = args
        self.pm = pm
        self.input_values_list = list()
        self.input_length_list = list()
        self.labels_list = list()
        self.regex = '[\,\?\.\!\-\;\:\"]'
        self.processor = processor
        self.text_list = text_list    # text_list, audio_list는 args.train_ver == "kspon" 일 경우에만 사용되는 값이다.
        self.audio_list = audio_list
        
    def train_dataset(self):
        if self.args.train_ver == "kss":
            kss_wavs = sorted(glob(os.path.join(self.cfg.kss_path, '**', '*.wav'), recursive=True))
            for wav_path in tqdm(kss_wavs):            
                audio, _ = librosa.load(wav_path, sr=16000)
                if self.args.vad == True:
                    non_silence_indices = del_noise(audio, top_db=30)
                    audio = np.concatenate([audio[start:end] for start, end in non_silence_indices])
                if audio.ndim > 1:
                    audio.np.delete(audio, -1, axis=1)
                    audio = audio.reshape(-1)
                input_value = self.processor(audio, sampling_rate=16000).input_values[0]
                self.input_values_list.append(input_value)
                self.input_length_list.append(len(input_value))
                txt_path = wav_path.replace(".wav", ".txt")
                with open(txt_path, 'r') as f:
                    text = f.read()
                text = re.sub(self.regex, '', text)
                with self.processor.as_target_processor():
                    self.labels_list.append(processor(text).input_ids)
        
        elif self.args.train_ver == "zeroth":
            zeorth_wavs = sorted(glob(os.path.join(self.cfg.train_zeroth_path, '**', '*.wav'), recursive=True))
            for wav_path in tqdm(zeorth_wavs):            
                audio, _ = librosa.load(wav_path, sr=16000)
                if self.args.vad == True:
                    non_silence_indices = del_noise(audio, top_db=30)
                    audio = np.concatenate([audio[start:end] for start, end in non_silence_indices])
                if audio.ndim > 1:
                    audio.np.delete(audio, -1, axis=1)
                    audio = audio.reshape(-1)
                input_value = self.processor(audio, sampling_rate=16000).input_values[0]
                self.input_values_list.append(input_value)
                self.input_length_list.append(len(input_value))
                txt_path = wav_path.replace(".wav", ".txt")
                with open(txt_path, 'r') as f:
                    text = f.read()
                text = re.sub(self.regex, '', text)
                with self.processor.as_target_processor():
                    self.labels_list.append(processor(text).input_ids)            
                                        
        elif self.args.train_ver == "kspon":
            for text, audio_path in zip(tqdm(text_list), audio_list):
                audio, _ = sf.read(audio_path)
                if self.args.vad == True:
                    non_silence_indices = del_noise(audio, top_db=30)
                    audio = np.concatenate([audio[start:end] for start, end in non_silence_indices])
                if audio.ndim > 1:
                    audio = np.delete(audio, 1, axis=1)
                    audio = audio.reshape(-1)
                input_value = processor(audio, sampling_rate=16000).input_values[0]
                self.input_values_list.append(input_value)
                self.input_length_list.append(len(input_value))
                with processor.as_target_processor():
                    self.labels_list.append(processor(text).input_ids)  
                
        return self.input_values_list, self.input_length_list, self.labels_list
    
    def train_test_split(self, input_values, input_lengths, labels):
        train_rate = self.pm.train_rate
        train_idx = int(train_rate * len(input_values))

        train_df = pd.DataFrame({'input_values': input_values[:train_idx], 
                                 'input_length': input_lengths[:train_idx], 
                                 'labels': labels[:train_idx]})
        test_df = pd.DataFrame({'input_values': input_values[train_idx:], 
                                'input_length': input_lengths[train_idx:], 
                                'labels': labels[train_idx:]})

        print("train dataset length: ", len(train_df))
        print("test dataset length: ", len(test_df))
        
        train_timit = Dataset.from_pandas(train_df)
        test_timit = Dataset.from_pandas(test_df)
        
        return train_timit, test_timit

    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
    
class Finetuning():
    def __init__(self, args, cfg, pm, processor):
        self.args = args
        self.cfg = cfg
        self.pm = pm
        self.processor = processor
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)
    
    def call_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53",
            gradient_checkpointing=True,
            ctc_loss_reduction="mean", 
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size = len(self.processor.tokenizer)
        )
    
        self.model.freeze_feature_encoder()
        
        print(self.model.config)
        print("=" * 100)
        print(self.model)
        
        return self.model
    
    def train(self, train_data, test_data):
        wandb.login()
        wandb.init(project=self.pm.pj_name)
        
        training_args = TrainingArguments(
              output_dir=self.cfg.output_dir,           
              group_by_length=self.pm.group_by_length,
              per_device_train_batch_size=self.args.batch_size,
              per_device_eval_batch_size=self.args.batch_size,
              evaluation_strategy=self.pm.evaluation_strategy,
              num_train_epochs=self.args.epochs,
              fp16=self.pm.fp16,
              gradient_checkpointing=self.pm.gradient_checkpointing,
              save_steps=self.args.eval_steps,
              eval_steps=self.args.eval_steps,
              logging_steps=self.args.eval_steps,
              learning_rate=self.pm.learning_rate,
              log_on_each_node=self.pm.log_on_each_node,
              weight_decay=self.pm.weight_decay,
              warmup_steps=self.pm.warmup_steps,
              eval_accumulation_steps=self.pm.eval_accumulation_steps,
              # fsdp=self.pm.fsdp,
              report_to=self.pm.report_to,
              save_total_limit=self.pm.save_total_limit,
        )
        
        trainer = Trainer(
            model=self.call_model(),
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.processor.feature_extractor,
        )
        try:
            trainer.train()
            
        except Exception as e:
            print(e)
            with open(os.path.join(self.cfg.output_dir, "train_log.txt"), 'w') as f:
                for obj in trainer.state.log_history:
                    f.write(obj)
            print("Train Log Save!")
        
        finally:
            print("Train Complete!!!")
            trainer.save_model(self.cfg.output_dir)
            print("Model & Processor Save!")

    
if __name__ == "__main__":
    from setproctitle import setproctitle
    setproctitle("Wav2Vec2 Finetuning Train")
    from config import Path_Config, Param_Config
    import argparse
    
    cfg = Path_Config()
    pm = Param_Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0,1,2,3", help="select cuda device number")
    parser.add_argument("--train_ver", type=str, default="kspon", help="select train type and dataset type")
    parser.add_argument("--vad", type=bool, default=False, help="decide to apply VAD")
    parser.add_argument("--batch_size", type=int, default=pm.batch_size)
    parser.add_argument("--epochs", type=int, default=pm.num_train_epochs)
    parser.add_argument("--eval_steps", type=int, default=pm.eval_steps, help="model evaluate & save step")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= args.device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)
    print('Count of using GPUs:', torch.cuda.device_count())
    
    vocab = Vocab(args, cfg, pm)
    vocab_path, text_list, audio_list = vocab.vocab_file_make()
    processor = processor(vocab_path)
    dataset = TrainDataset(args, pm, processor, text_list, audio_list)
    input_values_list, input_length_list, labels_list = dataset.train_dataset()
    train_timit, test_timit = dataset.train_test_split(input_values_list, input_length_list, labels_list)
    
    cer_metric = load_metric("cer")
    
    ft = Finetuning(args, cfg, pm, processor)
    
    ft.train(train_timit, test_timit)