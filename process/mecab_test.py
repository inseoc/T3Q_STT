from konlpy.tag import Mecab

if __name__=="__main__":
    mecab = Mecab()
    term = '뜨락또르'
    result = mecab.pos(term)
    print(result)
