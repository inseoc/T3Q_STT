import os
import shutil

from settings import WORK_DIR


def train(tm, *args, **kwargs):
    import torch
    from dataclasses import dataclass
    from typing import Dict, List, Optional, Union
    from transformers import Trainer, Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments
    from datasets import load_metric, Dataset
    import numpy as np
    import soundfile as sf

    MODEL_NAME=tm.param_info.get('model_name','default')
    MODEL_DIR=f'{WORK_DIR}/stt_data/models/transformers/{MODEL_NAME}'

    @dataclass
    class DataCollatorCTCWithPadding:

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )
    wer_metric = load_metric("wer")
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(f'{MODEL_DIR}/preprocessor_config.json'):
        processor=Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    else:
        processor = Wav2Vec2Processor.from_pretrained(f"./wav2vec2-large-xlsr-korean")
    if os.path.isfile(f'{MODEL_DIR}/pytorch_model.bin'):
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
    else:
        model = Wav2Vec2ForCTC.from_pretrained('./wav2vec2-large-xlsr-korean')
    if torch.cuda.is_available():
        model=model.to('cuda')
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def prepare_dataset(batch):
        if not batch['file_path'].endswith('.wav'):
            if not os.path.isfile(f'{batch["file_path"]}.wav'):
                os.system(f'ffmpeg -i {batch["file_path"]} -ar 16000 -f wav {batch["file_path"]}.wav')
            batch["file_path"]=f'{batch["file_path"]}.wav'
        audio, _ = sf.read(batch["file_path"])
        batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
        with processor.as_target_processor():
            txt_file=open(batch["text"], 'r', encoding='utf-8')
            batch["labels"] = processor(txt_file.read().strip()).input_ids
            txt_file.close()
        return batch

    audio_paths=[]
    script_paths=[]
    for audio_root, _, filenames in os.walk(tm.train_data_path):
        for filename in filenames:
            pure_filename = ".".join(filename.split(".")[:-1])
            if filename.endswith('.wav') or filename.endswith('.mp3') or filename.endswith('.mp4'):
                audio_path = f'{audio_root}/{filename}'
                target_audio_path = f'{audio_root}/{pure_filename}.wav'
                transcript_path = f'{audio_root}/{pure_filename}.txt'
                if not os.path.isfile(transcript_path):
                    continue
                if not os.path.isfile(target_audio_path):
                    os.system(f'ffmpeg -y -loglevel error -i "{audio_path}" -ar 16000 -f wav "{target_audio_path}"')

                audio_paths.append(target_audio_path)
                script_paths.append(transcript_path)
    print(script_paths)
    timit=Dataset.from_dict({'file_path':audio_paths, 'text':script_paths})
    train_dataset = timit.map(prepare_dataset, num_proc=4)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    model.save_pretrained(MODEL_DIR)

def init_svc(*args, **kwargs):
    return {}

def inference(df, params, *args, **kwargs):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from datasets import Dataset
    import soundfile as sf
    import torch

    if df.shape[0]>1 and df.shape[1]>0:
        MODEL_DIR=f'{WORK_DIR}/stt_data/models/transformers/{df.iloc[1][0]}'
    else:
        MODEL_DIR='./wav2vec2-large-xlsr-korean'

    if os.path.isfile(f'{MODEL_DIR}/preprocessor_config.json'):
        processor=Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    else:
        processor = Wav2Vec2Processor.from_pretrained(f"./wav2vec2-large-xlsr-korean")
    if os.path.isfile(f'{MODEL_DIR}/pytorch_model.bin'):
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
    else:
        model = Wav2Vec2ForCTC.from_pretrained('./wav2vec2-large-xlsr-korean')

    if torch.cuda.is_available():
        model=model.to('cuda')

    ds=Dataset.from_dict({'file_path':df.iloc[0]})
    test_ds=ds

    def map_to_array(batch):
        if not batch['file_path'].endswith('.wav'):
            if not os.path.isfile(f'{batch["file_path"]}.wav'):
                os.system(f'ffmpeg -i {batch["file_path"]} -ar 16000 -f wav {batch["file_path"]}.wav')
            batch["file_path"]=f'{batch["file_path"]}.wav'
        audio, _ = sf.read(batch["file_path"])
        batch["audio"] = audio
        return batch

    test_ds = test_ds.map(map_to_array)

    def map_to_pred(batch):
        inputs = processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.to("cuda")

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        batch["transcription"] = transcription
        return batch

    result = test_ds.map(map_to_pred, batched=True, batch_size=16, remove_columns=["audio"])

    print(result["transcription"])

if __name__=='__main__':
    from pandas import DataFrame
    class T3qModel:
        def __init__(self):
            self.param_info={
                'model_name':'모델명'
            }
            self.source_path=f'{WORK_DIR}/stt_data/train_sample/source_path'
            self.target_path=f'{WORK_DIR}/stt_data/train_sample/source_path'
            self.train_data_path=self.target_path
    t3q_model=T3qModel()
    # train(t3q_model)

    inference(DataFrame([[
        f'{WORK_DIR}/stt_data/train_sample/source_path/KsponSpeech_620003.wav',
    ],['모델명']]),None)