from setproctitle import *
setproctitle("STT_infer_test")

import os
import torch
import argparse
import torchaudio
import numpy as np
import torch.nn as nn
import Levenshtein as Lev

from torch import Tensor
from tqdm import tqdm
from kospeech.checkpoint import Checkpoint
from kospeech.data.audio.core import load_audio
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from symspellpy import SymSpell, Verbosity
from hangul_utils import split_syllables, join_jamos
from konlpy.tag import Mecab
# from kospeech.metrics import CharacterErrorRate


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
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


def _rm_dup_sentence(sentence):
    result = ''
    result += sentence[0]
    for i in range(1, len(sentence)):
        if sentence[i-1] != sentence[i]:
            result += sentence[i]

    return result


def symspell_sentence(mecab, symspell, sentence):
    result = list()
    sen_morphs = mecab.morphs(sentence)
    for morph in sen_morphs:
        term = split_syllables(morph)
        suggestions = symspell.lookup(term, Verbosity.ALL, max_edit_distance=1)
        if len(suggestions) == 0:
            result.append(morph + ' ')
        else:
            result.append(join_jamos(suggestions[0].term) + ' ')

    result_sen = ''.join(result)

    return result_sen


def sentence_infer(file, model, vocab, device):
    feature = parse_audio(file, del_silence=True)
    input_length = torch.LongTensor([len(feature)])

    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length)
    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())

    return sentence


def cal_cer(input_text, output_text):
    # cer_metric = CharacterErrorRate(vocab)
    input_text = input_text.replace(' ', '')
    output_text = output_text.replace(' ', '')

    dist = Lev.distance(output_text, input_text)
    length = len(input_text.replace(' ', ''))

    return dist / length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default='./')
    parser.add_argument('--audio_path', type=str, required=True, default='/kospeech/bin/lab_data/20170501_보도_1_.wav', help='input wav file dir path')
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--vocab_path', type=str, default='/kospeech/dataset/kspon/trainset/train_all_lables.csv')
    parser.add_argument('--rmdup', type=bool, default=False)
    parser.add_argument('--symspell', type=bool, default=False)
    parser.add_argument('--dict_path', default="/kospeech/dataset/kspon/vocab/vocab_5k_decomposed.txt", type=str, help='using dict path')
    parser.add_argument('--dist', default=1, type=int, help='set edit distance')
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(DEVICE)
    # print('Allocated cuda device', torch.cuda.current_device())

    vocab = KsponSpeechVocabulary(args.vocab_path)

    checkpoint = Checkpoint()
    resume_checkpoint = checkpoint.load(args.model_path)
    model = resume_checkpoint.model
    if isinstance(model, nn.DataParallel):
        model = model.module

    model.eval()

    symspell = SymSpell()
    mecab = Mecab()
    sentences = list()
    symspell.load_dictionary(args.dict_path, 0, 1)
    for wav_file in tqdm(sorted(os.listdir(args.audio_path)), desc='processing wav file'):
        if wav_file.endswith('.wav'):
            wav_file = os.path.join(args.audio_path, wav_file)
            sentence = sentence_infer(wav_file, model, vocab, DEVICE)
            if args.rmdup == True:
                if sentence[0] != '':
                    sentence = _rm_dup_sentence(sentence[0])
                    if args.symspell == True:
                        sentence = symspell_sentence(mecab, symspell, sentence)
                    sentences.append(sentence)
                else:
                    sentences.append(sentence[0])
            else:
                sentences.append(sentence[0])

    origin_txt = list()
    for txt_file in tqdm(sorted(os.listdir(args.audio_path)), desc='processing txt file'):
        if txt_file.endswith('.txt'):
            txt_file = os.path.join(args.audio_path, txt_file)
            with open(txt_file, 'r') as t:
                origin_data = t.read()
            origin_txt.append(origin_data)
    # breakpoint()
    if len(origin_txt) == len(sentences):
        total_cer = 0.0
        with open('/kospeech/bin/infer_output.txt', 'w') as f:
            for idx, sentence in enumerate(sentences):
                # cer, s1, s2 = cal_cer(origin_txt[idx], sentence, vocab)
                cer = cal_cer(origin_txt[idx], sentence)
                total_cer += cer
                # f.write(f'{origin_txt[idx]},{sentence},{cer}\n')
                f.write(f'{origin_txt[idx]},{sentence},{cer}\n')
        print("making output_text file complete")
        print(f"Total CER: {total_cer/len(sentences)}")
    else:
        print(f'origin data length: {len(origin_txt)}, output data length: {len(sentences)}')




if __name__=="__main__":
    main()
    print("complete!!!")
