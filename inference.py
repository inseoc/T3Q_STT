# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
stime = time.time()

import os
import argparse
from glob import glob

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import Levenshtein as Lev
import torchaudio
from torch import Tensor

# from symspellpy import SymSpell, Verbosity
from symspellpy_ko import KoSymSpell, Verbosity
from hangul_utils import split_syllables, join_jamos
from konlpy.tag import Mecab

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'wav') -> Tensor:
# def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


def rm_dup_sentence(sentence):
    result = ''
    result += sentence[0]
    for i in range(1, len(sentence)):
        if sentence[i-1] != sentence[i]:
            result += sentence[i]

    return result


def cal_cer(input_text, output_text):
    input_text = input_text.replace(' ', '').replace('\n', '')
    output_text = output_text.replace(' ', '').replace('\n', '')

    dist = Lev.distance(output_text, input_text)
    length = len(input_text.replace(' ', ''))

    return dist / length


def symspell_sentence(mecab, symspell, sentence):
    # result = list()
    # sen_morphs = mecab.morphs(sentence)
    # for morph in sen_morphs:
    #     term = split_syllables(morph)
    #     suggestions = symspell.lookup(term, Verbosity.ALL, max_edit_distance=1)
    #     if len(suggestions) == 0:
    #         result.append(morph + ' ')
    #     else:
    #         result.append(join_jamos(suggestions[0].term) + ' ')
    #
    # result_sen = ''.join(result)
    #
    # return result_sen
    for suggestion in symspell.lookup_compound(sentence, max_edit_distance=0):
        return suggestion.term

def main():
    parser = argparse.ArgumentParser(description='KoSpeech')
    parser.add_argument('--model_path', type=str, default='/kospeech/outputs/2022-04-18/14-06-01/4-model.pt')     # 한국어 stt best model
    # parser.add_argument('--model_path', type=str, default='/kospeech/dataset/kspon/outputs/2022-05-31/19-06-13/7-model.pt')     # 북한어 stt best model(0.28/no-eval)
    # parser.add_argument('--model_path', type=str, default='/kospeech/outputs/2022-06-13/15-00-32/7-model.pt')     # 북한어 stt best model(0.33/eval)
    parser.add_argument('--audio_path', type=str, required=True, default='/kospeech/dataset/kspon/test_dataset/ko_testset')
    parser.add_argument('--device', type=str, default=3)
    parser.add_argument('--vocab_path', type=str, default='/kospeech/dataset/kspon/trainset/train_all_lables.csv')
    # parser.add_argument('--dict_path', type=str, default='/kospeech/dataset/kspon/vocab/vocab_5k_decomposed.txt')
    parser.add_argument('--rm_dup', type=bool, default=False)
    parser.add_argument('--symspell', type=bool, default=False)
    opt = parser.parse_args()

    # GPU allocation
    DEVICE = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(DEVICE)
    print('Allocated cuda device', torch.cuda.current_device())

    # SymSpell Option
    # symspell = SymSpell()
    symspell = KoSymSpell()
    mecab = Mecab()
    # symspell.load_dictionary(opt.dict_path, 0, 1)
    symspell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)

    if os.path.isfile(opt.audio_path):
        audio_paths = [opt.audio_path]
        text_paths = [opt.audio_path.replace('.wav', '.txt')]
    elif os.path.isdir(opt.audio_path):
        data_paths = glob(os.path.join(opt.audio_path, '*'))
        audio_paths = sorted([audio for audio in data_paths if audio.endswith(".wav")])
        text_paths = sorted([text for text in data_paths if text.endswith(".txt")])
    # print(len(audio_paths), len(text_paths))

    total_cer = 0.0
    with open('/kospeech/dataset/kspon/test_dataset/infer_result.txt', 'w') as f:
        for idx, audio_path in enumerate(tqdm(audio_paths)):
            re_sentence = ''

            with open(text_paths[idx], 'r') as t:
                text = t.read()

            feature = parse_audio(audio_path, del_silence=True)
            input_length = torch.LongTensor([len(feature)])
            vocab = KsponSpeechVocabulary(opt.vocab_path)

            model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(DEVICE)
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.eval()

            if isinstance(model, ListenAttendSpell):
                model.encoder.device = DEVICE
                model.decoder.device = DEVICE
                y_hats = model.recognize(feature.unsqueeze(0), input_length)
            elif isinstance(model, DeepSpeech2):
                model.device = DEVICE
                y_hats = model.recognize(feature.unsqueeze(0).to(DEVICE), input_length)
            elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
                y_hats = model.recognize(feature.unsqueeze(0), input_length)

            sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
            # breakpoint()
            if opt.rm_dup == True:
                sentence = rm_dup_sentence(sentence[0])
                if opt.symspell == True:
                    sentence = symspell_sentence(mecab, symspell, sentence)
                cer = cal_cer(text, sentence)
                f.write(f'{text}, {sentence}\n')
            else:
                cer = cal_cer(text, sentence[0])
                f.write(f'{text}, {sentence[0]}\n')

            total_cer += cer

    stt_cer = total_cer/len(audio_paths)
    print(f'CER: {stt_cer}')

if __name__=="__main__":
    main()
    print("working time!!!", time.time() - stime)
