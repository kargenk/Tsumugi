#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from time import sleep
import re

def scraping_from_url(url):
    """
    uta-netのURLから楽曲情報をスクレイピングする関数．
    
    Parameters
    ----------
    url : str
        uta-netのURL
    
    Returns
    ----------
    soup : object
        スクレイピング結果を格納した，BeautifulSoupオブジェクト
    """
    sleep(1)  # webサイトに負担をかけないように間隔をあける
    
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    
    return soup

def create_songs_dataframe(url):
    """
    URLから楽曲のデータフレームを作成する関数．
    
    Parameters
    ----------
    url : str
        uta-netのURL
    
    Returns
    ----------
    song_df : pd.DataFrame
        楽曲情報のデータフレーム
    """
    soup = scraping_from_url(url)
    
    # htmlをパースして曲名，楽曲のURL，アーティスト名，作詞，作曲者名を取得
    contents = []
    contents.append(soup.find_all(href=re.compile('/song/\d+/$')))  # 楽曲URL用
    contents.append(soup.find_all(href=re.compile('/song/\d+/$')))  # 楽曲名用
    contents.append(soup.find_all(class_=re.compile('td2')))        # アーティスト名
    contents.append(soup.find_all(class_=re.compile('td3')))        # 作詞名
    contents.append(soup.find_all(class_=re.compile('td4')))        # 作曲者名
    
    # 各情報をリストに格納
    informations = []
    for i, content in enumerate(contents):
        tmp_list = []
        for element in content:
            if i == 0:
                tmp_list.append(element.get('href'))  # 楽曲URL
            else:
                tmp_list.append(element.string)
        informations.append(tmp_list)

    # 楽曲情報データフレームの作成
    songs_df = pd.DataFrame({
        'URL': informations[0],
        'SongName': informations[1],
        'Artist': informations[2],
        'Lyricist': informations[3],
        'Composer': informations[4]
    })
    
    # URLにホスト名を追加
    songs_df.URL = songs_df.URL.apply(lambda x : 'https://www.uta-net.com' + x)
    
    # 歌詞を追加
    lyrics = []
    for url in songs_df.URL:
        page = (scraping_from_url(url))
        lyrics.append(page.find(id='kashi_area').text)
    songs_df['Lyric'] = lyrics
    
    return songs_df

def remove_fullspace_and_newline(text):
    """
    全角スペース(\u3000)と改行(\n)を除去する関数．
    
    Parameters
    ----------
    text : str
        前処理を行う前の歌詞
    
    Returns
    ----------
    text_removed : str
        全角スペースと改行を除去した歌詞
    """
    text_removed = text.replace('\u3000', '')      # 全角スペース(\u3000)の除去
    text_removed = text_removed.replace('\n', '')  # 改行(\n)の除去
    
    return text_removed

def reweight_distribution(original_distribution, temperature=0.5):
    """
    softmax temperature: 元の確率分布から，新しい確率分布を生成する関数．

    Parameters
    ----------
    original_distribution : numpy.ndarray, dim=1
        モデルのソフトマックス関数の出力である元の確率分布
    temperature : float, [0.0, 1.0]
        出力分布のエントロピーを定量化する係数，
        次の文字の選択をどれくらい意外なものにするのかを制御するパラメータ，0.0がgreedy，1.0がランダム
    
    Returns
    ----------
    new_distribution : object
        元の確率分布から再荷重された新しい確率分布
    """
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    new_distribution = distribution / np.sum(distribution)

    return new_distribution

def sample(preds, temperature=1.0):
    """
    モデルから得られた確率分布を再荷重し，そこから次の単語のインデックスをサンプリングする関数．
    
    Parameters
    ----------
    preds : numpy.ndarray, dim=1
        モデルのソフトマックス関数の出力である元の確率分布
    temperature : float, [0.0, 1.0]
        出力分布のエントロピーを定量化する係数，
        次の文字の選択をどれくらい意外なものにするのかを制御するパラメータ，0.0がgreedy，1.0がランダム
    
    Returns
    ----------
    next_id : int
        次の単語のインデックス
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)  # 引数は(試行回数，確率分布，サイズ)
    next_id = np.argmax(probas)

    return next_id
