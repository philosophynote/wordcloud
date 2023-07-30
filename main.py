import MeCab
import numpy as np
import pandas as pd
import nlplot
import plotly
# 半角⇄全角の変換を実行する
import mojimoji
# 正規化処理
import neologdn
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt
import re
import itertools

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_colwidth', 5000)

# 前処理に必要な関数
# 数字、アルファベットを半角に変換し、正規化する
def normalize_text(text):
    result = mojimoji.zen_to_han(text, kana=False)
    result = neologdn.normalize(result)
    return result

def text_to_only_noun(text):
    with open("./Japanese.txt","r") as f:
        stopwords = f.read().split("\n")
    m = MeCab.Tagger('-d /Users/naokitakahashi/opt/anaconda3/envs/naokitakahashi/lib/mecab/dic/mecab-ipadic-neologd')
    m.parse('')
    pattern = r"^\d.*"
    text = normalize_text(text)
    m_text = m.parse(text)
    basic_words = []
    #mecabの出力結果を単語ごとにリスト化
    m_text = m_text.split('\n')
    for row in m_text:
        #Tab区切りで形態素、その品詞等の内容と分かれているので単語部のみ取得
        word = row.split("\t")[0]
        #最終行はEOS
        if word == 'EOS':
            break
        else:
            pos = row.split('\t')[1]
            slice_ = pos.split(',')
            #品詞を取得する
            parts = slice_[0]
            if parts == '記号':
                if word != '。':
                    continue

                #読点のみ残す
                # basic_words.append(word)
            #活用語の場合は活用指定ない原型を取得する。
            # elif slice_[0] in ('形容詞', '動詞'):
            #         basic_words.append(slice_[-3])

            #活用しない語についてはそのままの語を取得する
            elif slice_[0] in ('名詞'):
                basic_words.append(word)

    basic_words_excluded = [basic_word for basic_word in basic_words if basic_word  not in stopwords]
    basic_unique_words = set(basic_words_excluded)
    basic_words_joined = ' '.join(list(basic_unique_words))
    return basic_words_joined

def text_to_words(text):
    with open("./Japanese.txt","r") as f:
        stopwords = f.read().split("\n")
    m = MeCab.Tagger('-d /Users/naokitakahashi/opt/anaconda3/envs/naokitakahashi/lib/mecab/dic/mecab-ipadic-neologd')
    m.parse('')
    pattern = r"^\d.*"
    text = normalize_text(text)
    m_text = m.parse(text)
    basic_words = []
    #mecabの出力結果を単語ごとにリスト化
    m_text = m_text.split('\n')
    for row in m_text:
        #Tab区切りで形態素、その品詞等の内容と分かれているので単語部のみ取得
        word = row.split("\t")[0]
        #最終行はEOS
        if word == 'EOS':
            break
        else:
            pos = row.split('\t')[1]
            slice_ = pos.split(',')
            #品詞を取得する
            parts = slice_[0]
            if parts == '記号':
                if word != '。':
                    continue

                #読点のみ残す
                # basic_words.append(word)
            #活用語の場合は活用指定ない原型を取得する。
            elif slice_[0] in ('動詞'):
                    basic_words.append(slice_[-3])

            #活用しない語についてはそのままの語を取得する
            elif slice_[0] in ('名詞','副詞'):
                basic_words.append(word)

    basic_words_excluded = [basic_word for basic_word in basic_words if basic_word  not in stopwords]
    basic_unique_words = set(basic_words_excluded)
    basic_words_joined = ' '.join(list(basic_unique_words))
    return basic_words_joined
  
# csvファイルの読み込み
df = pd.read_csv('sample.csv')
df["token_oniy_noun"] = df["メモ"].apply(text_to_only_noun)
npt_only_noun = nlplot.NLPlot(df, target_col="token_oniy_noun")

# 単語の出現頻度を棒グラフで表示
fig_unigram_only_noun = npt_only_noun.bar_ngram(
    title="",
    xaxis_label='word_count',
    yaxis_label='word',
    ngram=1,
    top_n=50,
    width=800,
    height=1100,
    verbose=False,
)

fig_unigram_only_noun.show()


fig_wc_only_noun = npt_only_noun.wordcloud(
    width=640,
    height=480,
    max_words=50,
    max_font_size=200,
    colormap='tab20_r',
    mask_file=None,
    save=False,
)

plt.figure()
plt.imshow(fig_wc_only_noun, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_2022年上半期（名詞のみ）.png')