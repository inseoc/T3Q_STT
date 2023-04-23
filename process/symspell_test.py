
'''
기본 symspell 및 hangul_utils 라이브러리를 활용한 후처리 코드
'''
import argparse

from symspellpy import SymSpell, Verbosity
from hangul_utils import split_syllables, join_jamos
from konlpy.tag import Mecab


def _rm_dup_sentence(sentence):
    result = ''
    result += sentence[0]
    for i in range(1, len(sentence)):
        if sentence[i-1] != sentence[i]:
            result += sentence[i]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path', default='./', type=str, help='using dict path')
    parser.add_argument('--dist', default=1, type=int, help='set edit distance')
    args = parser.parse_args()

    symspell = SymSpell()
    mecab = Mecab()
    result = list()

    dictionary_path = args.dict_path
    # dictionary_path = "/kospeech/dataset/kspon/vocab/vocab_5k_decomposed.txt"
    # dictionary_path = "/kospeech/dataset/kspon/vocab/vocab_5k_space_decomposed.txt"
    symspell.load_dictionary(dictionary_path, 0, 1)

    # sentence = "공공을  수로서는 기기자회견문을  통해서  지난  적폐정부가 임명한  공공 기관 관력배들이 비정규지의 정기지기를의 전환을  반대해 나서고 있있다고  규탄했습니다"
    # "중국의 신화 통신 중앙 텔레비죤 방송 중국 보도망 청년망 홍콩 명보 일본의 교도통신 니혼게이자이 신붕 마이니찌 신붕 엔에이치케이 방송 러시아의 따스 통신 인터네트 신문 렌따루 러시아 인터네트 잡지 림치치 인터네트 홈페이지 빠뜨리오띠 모스크바 체스꼬의 체떼까 통신 프랑스의 에이에프피 통신 도이췰란드의 데페아 통신 이딸리아 신문 라레뿌블리까 영국의 비비씨 방송 신문 데일리 메일 베네수엘라의 뗄레스 루 텔레비죤 방송 도이 텔레비죤 방송 브라질 신문 아구와 베르데 미국의 에이피 통신 유피아이 통신 신문 뉴욕 타임즈 오스트랄리아의 에이비씨 방송과 나이제리아 김일성 김정일주의연구 전국위원회 백두산 체스꼬 조선친선협회 뽈스까 조선민주주의인민공화국 탐구를 위한 국제친선발기 인터네트 홈페이지들을 비롯한 여러 나라 출판보도물들이 조선 외무성 대변인 유엔안전보이사회 제재결의를 전면배격 유엔안보이사회 결의는 전쟁행위 북조선 핵억제력을 강화할 것이라고 천명 등의 제목으로 우리나라 외무성 대변인 성명을 전문 또는 요지로 보도했습니다"
    sentence = "중국의  신화 통신  중앙 텔레비죤  방송  중국  보도망  청연망 홍콩콩 영보  일일본의  교도통신  도꾜신신붕  니기이자이 신붕  마이니주 신붕  엔네치케이  방송  러시아의  따스스 통신  인터네트  신문  왼따르  인터네트 잡지 림치  인터네트  홈페이이지  빠뜨리오띠  모스크바  체스꼬의 체택가 통신  프랑스의  에이에프피 통신  도이이췰란드의  대페아 통신  이딸리아아 신문라  레프리카 영국의  비비씨 방송  신문 일린비일  베네수수엘엘라의   뗄레쓰루  텔레비죤 방송  브이텔레비죤  방송  브라질질신문  아오와  베르데  미국의  에이피 통이신 유피아이통이신 신문 뉴욕 타임즈  오스스트랄리아아의  에비씨 방송과  나이지리아아  김일성 김정일주연연구 전국위원회 백두산 체스꼬  조선친선협회 뽈스까 조선민주주주의의인인민공화국  탐구를 위한 국제친선발계  인터네트 홈페이지들을  비롯한  여러 나라  출출판보도물들이  조선  외무성 대변인  유엔안보이사회 제재결의를 전면 배격  유엔안보이사회  결의는  전쟁행위위 북조선  핵억제력을 강화할할 것이이라고 성명  등의  제목으로  우리나라  외무성 대변인 성명을  전문 또는  요지로  보도했습습니다"
    sentence = _rm_dup_sentence(sentence)

    sen_morphs = mecab.morphs(sentence)
    for morph in sen_morphs:
        term = split_syllables(morph)
        suggestions = symspell.lookup(term, Verbosity.ALL, max_edit_distance=args.dist)
        if len(suggestions) == 0:
            result.append(morph + ' ')
        else:
            result.append(join_jamos(suggestions[0].term) + ' ')

    result_sen = ''.join(result)
    print(result_sen)


'''
symspellpy_ko 라이브러리를 변형한 후처리 코드
'''
# import argparse
# from ko_sym_spell import KoSymSpell, Verbosity
# from konlpy.tag import Mecab
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--vocab_path', default='./', type=str, help='using vocab path')
#     parser.add_argument('--bigram_path', default='./', type=str, help='using bigram vocab path')
#     args = parser.parse_args()
#
#     symspell = KoSymSpell()
#     mecab = Mecab()
#     symspell.load_korean_dictionary(args.vocab_path, args.bigram_path, decompose_korean=True, load_bigrams=True)
#
#     sentence = '공공을 수로서는 기기자회견문을 통해서 지난 적폐정부가 임명한 공공 기관 관력배들이 비정규지의 정기지기를의 전환을 반대해 나서고 있있다고 규탄했습니다'
#     sen_morphs = mecab.morphs(sentence)
#
#     for morph in sen_morphs:
#         # term = split_syllables(morph)
#         suggestions = symspell.lookup_compound(morph, Verbosity.ALL, max_edit_distance=2)
#         for suggestion in suggestions:
#             print(suggestion.term, suggestion.distance, suggestion.count)


if __name__=="__main__":
    main()
    print("Complete!")
