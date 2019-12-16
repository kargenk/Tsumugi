#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import re

def scraping_from_url(url):
    """
    uta-netのURLから楽曲情報をスクレイピングする関数．
    
    Parameters
    ----------
    url : str
        uta-netのURL
    
    returns
    ----------
    soup : object
        スクレイピング結果を格納した，BeautifulSoupオブジェクト
    """
    sleep(3)  # webサイトに負担をかけないように間隔をあける
    
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
    
    returns
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
