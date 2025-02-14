#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# We use the Japanese model, but translations are available in other languages. For more information,
# please read the Llama3-related documentation on META.
# Copyright(C) Omicronware.
import re
from llama_cpp import Llama


llm = Llama(
    # https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF
    model_path="models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    chat_format="llama-3",
    n_gpu_layers=33,
    seed=1337,
    n_ctx=512,
    temperature=0.4,
    top_p = 0.8,
    streaming = True,
    verbose=False
)

def llama(from_lang, to_lang, transcribed_text):
    
    from_language = languages_dict(from_lang.lower())['ja']
    to_language   = languages_dict(to_lang.lower())['ja']
    
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": f"あなたは優秀な翻訳者です。{from_language}を{to_language}に翻訳してください。ただし、翻訳結果のみを出力してください。",
            },
            {
                "role": "user",
                "content": f"{transcribed_text}",
            },
        ],
        max_tokens=512,
    )
    
    result = response["choices"][0]["message"]["content"].strip()
    
    # "Here is the translation:" というフレーズを削除（改行・スペースを含む可能性を考慮）
    result = re.sub(r"Here is the translation:\s*", "", result)

    return result.strip()


def languages_dict(lang_code): #from ISO639-1 to Japanese or English

    LANGUAGES = {
        'en': {'ja': "英語", 'en': "English"},
        'ja': {'ja': "日本語", 'en': "Japanese"},
        'zh-cn': {'ja': "中国語（簡体語）", 'en': "Chinese (Simplified)"},
        'zh-tw': {'ja': "中国語（繁体語）", 'en': "Chinese (Traditional)"},
        'zh': {'ja': "中国語（簡体語）", 'en': "Chinese"},
        'ko': {'ja': "韓国語", 'en': "Korean"},
        'es': {'ja': "スペイン語", 'en': "Spanish"},
        'fr': {'ja': "フランス語", 'en': "French"},
        'de': {'ja': "ドイツ語", 'en': "German"},
        'it': {'ja': "イタリア語", 'en': "Italian"},
        'pt': {'ja': "ポルトガル語", 'en': "Portuguese"},
        'nl': {'ja': "オランダ語", 'en': "Dutch"},
        'ru': {'ja': "ロシア語", 'en': "Russian"},
        'ar': {'ja': "アラビア語（標準）", 'en': "Arabic (Standard)"},
        'hi': {'ja': "ヒンディー語", 'en': "Hindi"},
        'bn': {'ja': "ベンガル語", 'en': "Bengali"},
        'ur': {'ja': "ウルドゥー語", 'en': "Urdu"},
        'th': {'ja': "タイ語", 'en': "Thai"},
        'mn': {'ja': "モンゴル語", 'en': "Mongolian"},
        'sv': {'ja': "スウェーデン語", 'en': "Swedish"},
        'no': {'ja': "ノルウェー語", 'en': "Norwegian"},
        'fi': {'ja': "フィンランド語", 'en': "Finnish"},
        'he': {'ja': "ヘブライ語", 'en': "Hebrew"},
        'uk': {'ja': "ウクライナ語", 'en': "Ukrainian"},
        'auto':{'ja': "自動検出した言語", 'en':"Auto-detected language"}
    }


    # キーが存在しない場合のデフォルト
    return LANGUAGES.get(lang_code.lower(), {'ja': "英語", 'en': "English"})



# 実行例
if __name__ == "__main__":
  
  transcribed_text = "昨日、2次会が長引き終電に乗り遅れた。仕方がないのでカラオケボックスで一夜を明かした。"
  
  from_lang = 'ja'
  to_lang = 'en'
  print(llama(from_lang, to_lang, transcribed_text))
