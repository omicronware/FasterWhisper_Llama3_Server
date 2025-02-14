#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This is a sample code for a speech translation server that runs completely locally and
# can be used as is if your environment is equipped with NVIDIA's GPU/CUDA. 
# Multi-client and HTTP/HTTPS are supported. Some error messages are output as is.
# Please use this server model in accordance with each LICENSE.
# Copyright(C) Omicronware.
import os
import sys
import tempfile
import traceback
import ctypes

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Llama build & install on Windows/Poweshell もし手動でインストールしたい場合
# $Env:FORCE_CMAKE = 1
# $Env:CMAKE_ARGS = "-DLLAMA_CUDA=ON"
# pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
import llama

# gevent を利用した WSGI サーバー
# pip install gevent
from gevent.pywsgi import WSGIServer
import gevent


app = Flask(__name__)

def is_cuda_available():
    """Windows + nvcuda.dll ベースで CUDA の利用可能性を確認するサンプル実装"""
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.kernel32.LoadLibraryW("nvcuda.dll"))
    except OSError:
        return False

MODEL_NAME = "large-v3-turbo"
DEVICE = "cuda" if is_cuda_available() else "cpu"
COMPUTE_TYPE = "auto"  # 例: "float16" 等にするとGPUメモリ削減

try:
    print(f"Loading faster-whisper model '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE}) ...")
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    batched_model = BatchedInferencePipeline(model=model)
    print("Batched Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e, file=sys.stderr)
    sys.exit(1)


@app.errorhandler(Exception)
def handle_exception(e):
    """Flask の全ての未処理例外をキャッチする"""
    tb = traceback.format_exc()
    return jsonify({
        "error": "内部サーバーエラーが発生しました。",
        "details": str(e),
        "trace": tb
    }), 500

@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    """
    GET:
      - { "status": "ok" } を返してヘルスチェック (HTTP/HTTPSどちらでも可)

    POST:
      - フォームデータ:
        - audio_file (必須): mp3 等の音声ファイル
        - from_language (任意): Faster-Whisper の認識言語 (未指定で自動判定)
        - to_language (任意): 翻訳先言語 (例: 'en', 'ja')
      - JSON 形式で文字起こし・翻訳結果を返す
    """
    if request.method == 'GET':
        return jsonify({"status": "ok"}), 200

    if 'audio_file' not in request.files:
        return jsonify({"error": "audio_file がリクエストに含まれていません"}), 400

    audio_file = request.files['audio_file']
    from_language = request.form.get("from_language", None)
    to_language   = request.form.get("to_language", None)

    # 一時ファイルに音声を保存
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_file.save(tmp)
            tmp_filename = tmp.name
    except Exception as e:
        return jsonify({
            "error": "音声ファイルの一時保存に失敗しました",
            "details": str(e)
        }), 500

    try:
        # 文字起こし
        segments, info = batched_model.transcribe(tmp_filename, language=from_language)
        full_text = "".join(segment.text for segment in segments)
        segments_list = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]

        detected_language = getattr(info, "language", from_language)

        # 翻訳が必要なら実行
        translated_text = None
        if to_language:
            from_language = detected_language if detected_language else "auto"
            translated_text = llama.llama(from_language, to_language, full_text)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({
            "error": "文字起こし処理に失敗しました",
            "details": str(e),
            "trace": tb
        }), 500
    finally:
        # 一時ファイル削除
        try:
            os.remove(tmp_filename)
        except Exception:
            pass

    # 返却JSON
    result = {
        "transcript_text": full_text,
        "translated_text": translated_text,
        "segments": segments_list,
        "language": detected_language
    }
    return jsonify(result), 200


if __name__ == '__main__':
    # 証明書ファイル (自己署名や正式証明書など)
    SSL_CERT_FILE = "server.crt"
    SSL_KEY_FILE = "server.key"

    # HTTP用 WSGIServer (0.0.0.0:9000)
    http_server = WSGIServer(('0.0.0.0', 9000), app)

    # HTTPS用 WSGIServer (0.0.0.0:9443)
    # certfile/keyfile を指定する
    try:
      https_server = WSGIServer(
          ('0.0.0.0', 9443), 
          app,
          keyfile=SSL_KEY_FILE,
          certfile=SSL_CERT_FILE
      )

      print("Starting HTTP server on port 9000")
      http_server.start()

      print("Starting HTTPS server on port 9443")
      https_server.start()

      print("Servers are running. Press Ctrl+C to stop.")
      
    except ssl.SSLError as e:
        print(f"SSLエラー: {e}", file=sys.stderr)
    except Exception as e:
        print(f"サーバーの起動時にエラーが発生しました: {e}", file=sys.stderr)
        traceback.print_exc()


    try:
        # gevent で並行実行 (Ctrl+C で停止)
        gevent.wait()
    except KeyboardInterrupt:
        print("\nShutdown requested. Stopping servers...")
        http_server.stop()
        https_server.stop()
        print("Servers stopped successfully.")
        sys.exit(0)