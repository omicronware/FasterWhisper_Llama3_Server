# FasterWhisper & Llama-3-ELYZA-JP-8B Translation Server

このプロジェクトは、ローカル環境で動作する音声文字起こし＆翻訳サーバーです。Faster-Whisperを用いて音声認識を行い、Llama-3-ELYZA-JP-8B を利用して翻訳を行います。

## 特徴
- **完全ローカル動作**: 外部APIを利用せず、プライバシーを確保。
- **GPU対応**: CUDA 12 & cuDNN 9.6.x を推奨。
- **マルチランゲージ対応**: 日本語を含む多言語対応。

## インストール
### 依存ライブラリのインストール
```sh
pip install -r requirements.txt
```

### Llama-3-ELYZA-JP-8B のダウンロード
モデルのダウンロードは以下のURLから行ってください。
- [ELYZA Model Download](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/tree/main)

ダウンロード後、モデルを `models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf` に配置してください。

## 使用方法
### サーバーの起動
```sh
python fasterwhisper_llama_server.py
```

サーバーが起動すると、以下のエンドポイントが利用可能になります。
- **ヘルスチェック**: `GET /transcribe`
- **音声文字起こし＆翻訳**: `POST /transcribe`
  - `audio_file`: 音声ファイル（mp3, wav など）
  - `from_language`: 入力言語（オプション、未指定で自動検出）
  - `to_language`: 翻訳先言語（例: 'en', 'ja'）

リクエスト例（`cURL` 使用）:
```sh
curl -X POST http://localhost:9000/transcribe \
     -F "audio_file=@sample.mp3" \
     -F "from_language=ja" \
     -F "to_language=en"
```

レスポンス例:
```json
{
  "transcript_text": "昨日、2次会が長引き終電に乗り遅れた。",
  "translated_text": "Yesterday, the afterparty lasted too long and I missed the last train.",
  "segments": [
    {"start": 0.0, "end": 3.5, "text": "昨日、2次会が長引き終電に乗り遅れた。"}
  ],
  "language": "ja"
}
```

## 環境要件
- **OS**: Windows 10 Pro 22H2 以上（Linux も可）
- **CUDA**: 12 以上（推奨）
- **cuDNN**: 9.6.x 以上（推奨）
- **Python**: 3.9 以上
- **GPU**: NVIDIA GPU（推奨）

## ライセンス
- **Faster-Whisper**: [MIT License](https://github.com/ggerganov/whisper.cpp/blob/main/LICENSE)
- **Llama-3-ELYZA-JP-8B**: [ELYZA License](https://huggingface.co/elyza/llama-3-elyza-jp-8b)
- **このリポジトリのソースコード**: MIT License

## 参考
このプロジェクトは以下を参考に作成されました。
- [FasterWhisper_Googletrans_Server](https://github.com/omicronware/FasterWhisper_Googletrans_Server/)

