#!/usr/bin/env python3
"""
jt-live-whisper 遠端 Whisper ASR 伺服器
部署到 GPU 伺服器，提供 REST API 讓本機上傳音訊檔進行語音辨識。

後端引擎自動偵測：
  1. faster-whisper (CTranslate2 CUDA) — x86_64 GPU，速度最快
  2. openai-whisper (PyTorch CUDA) — aarch64 GPU（如 DGX Spark），也能 GPU 加速
  3. faster-whisper (CPU) — 無 GPU 降級

依賴：faster-whisper, fastapi, uvicorn, python-multipart
      （aarch64 無 CTranslate2 CUDA 時額外需要 openai-whisper）
      （講者辨識需額外安裝 resemblyzer, spectralcluster）
啟動：python3 server.py [--port 8978] [--host 0.0.0.0]

Author: Jason Cheng (Jason Tools)
"""

import argparse
import asyncio
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time

# 原始碼編譯的 CTranslate2 將 libctranslate2.so 安裝到 /usr/local/lib
# 需在 import ctranslate2 前確保 LD_LIBRARY_PATH 包含此路徑
if "/usr/local/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="jt-whisper-server")

# ── 作業追蹤 ──
_active_task_lock = threading.Lock()
_active_task = None  # dict: {type, model, language, started, client_ip} or None

def _set_active_task(task_type, model, language, client_ip=""):
    global _active_task
    with _active_task_lock:
        _active_task = {
            "type": task_type,
            "model": model,
            "language": language,
            "started": time.time(),
            "client_ip": client_ip,
        }

def _clear_active_task():
    global _active_task
    with _active_task_lock:
        _active_task = None

def _get_active_task():
    with _active_task_lock:
        if _active_task is None:
            return None
        return dict(_active_task)

# ── 偵測最佳後端引擎 ──
_models: dict = {}
_backend = "faster-whisper"  # "faster-whisper" 或 "openai-whisper"
_device = "cpu"
_compute_type = "int8"
_torch_device = "cpu"

if torch.cuda.is_available():
    _torch_device = "cuda"
    # 嘗試 CTranslate2 CUDA（faster-whisper 用）
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            _device = "cuda"
            _compute_type = "float16"
            _backend = "faster-whisper"
            print("[引擎] faster-whisper (CTranslate2 CUDA)")
        else:
            raise RuntimeError("CTranslate2 無 CUDA")
    except Exception:
        # CTranslate2 沒 CUDA，嘗試 openai-whisper（PyTorch CUDA）
        try:
            import whisper as openai_whisper  # noqa: F401
            _backend = "openai-whisper"
            _device = "cuda"
            print("[引擎] openai-whisper (PyTorch CUDA)")
        except ImportError:
            print("[警告] CTranslate2 無 CUDA 且 openai-whisper 未安裝，改用 CPU")
            _backend = "faster-whisper"
else:
    print("[引擎] faster-whisper (CPU)")

# ── 偵測 diarization 套件 ──
_HAS_DIARIZE = False
try:
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings("ignore", message="pkg_resources is deprecated")
        from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: F401
    from spectralcluster import SpectralClusterer  # noqa: F401
    from spectralcluster import refinement  # noqa: F401
    _HAS_DIARIZE = True
    print(f"[講者辨識] resemblyzer + spectralcluster 可用 (device={_torch_device})")
except ImportError:
    print("[講者辨識] resemblyzer/spectralcluster 未安裝，diarize API 停用")


# ── Diarization 核心函式 ──

def _diarize(wav_path, segments, num_speakers=None):
    """用 resemblyzer + spectralcluster 辨識講者。
    segments: list of dict，每個含 start, end, text
    回傳: list of int（講者編號 0-based），失敗回傳 None
    """
    from collections import Counter
    from resemblyzer import VoiceEncoder, preprocess_wav
    from spectralcluster import SpectralClusterer, refinement

    if not segments:
        return None

    # 載入音訊
    wav = preprocess_wav(wav_path)
    sr = 16000  # resemblyzer preprocess_wav 輸出 16kHz

    # 初始化聲紋編碼器（有 GPU 就用 GPU）
    encoder = VoiceEncoder(_torch_device)
    print(f"[diarize] 提取聲紋（{len(segments)} 段, device={_torch_device}）")

    # ── 合併連續短段落（< 0.8s）再提取 embedding ──
    merge_groups = []
    i = 0
    while i < len(segments):
        duration = segments[i]["end"] - segments[i]["start"]
        if duration < 0.8:
            group = [i]
            j = i + 1
            while j < len(segments) and (segments[j]["end"] - segments[j]["start"]) < 0.8:
                group.append(j)
                j += 1
            if len(group) > 1:
                merge_groups.append(group)
                i = j
                continue
        i += 1
    merged_set = set()
    merged_emb_map = {}
    for group in merge_groups:
        combined_audio = np.concatenate([
            wav[int(segments[idx]["start"] * sr):int(segments[idx]["end"] * sr)]
            for idx in group
        ])
        if len(combined_audio) >= int(0.3 * sr):
            try:
                emb = encoder.embed_utterance(combined_audio)
                for idx in group:
                    merged_emb_map[idx] = emb
                    merged_set.add(idx)
            except Exception:
                pass

    # 逐段提取聲紋
    embeddings = []
    valid_indices = []

    for i, seg in enumerate(segments):
        if i in merged_emb_map:
            embeddings.append(merged_emb_map[i])
            valid_indices.append(i)
            continue

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        duration = seg["end"] - seg["start"]
        if duration < 0.5:
            mid = (seg["start"] + seg["end"]) / 2
            start_sample = max(0, int((mid - 0.25) * sr))
            end_sample = min(len(wav), int((mid + 0.25) * sr))

        audio_slice = wav[start_sample:end_sample]

        if len(audio_slice) < int(0.3 * sr):
            embeddings.append(None)
            continue

        try:
            if duration >= 1.6:
                emb, partials, _ = encoder.embed_utterance(
                    audio_slice, return_partials=True, rate=1.6, min_coverage=0.75
                )
                emb = np.median(partials, axis=0)
                emb = emb / np.linalg.norm(emb)
            else:
                emb = encoder.embed_utterance(audio_slice)
            embeddings.append(emb)
            valid_indices.append(i)
        except Exception:
            embeddings.append(None)

    if not valid_indices:
        print("[diarize] 無法提取任何有效聲紋")
        return None

    print(f"[diarize] 分群辨識（{len(valid_indices)} 有效段落）")

    # 組合有效 embedding 矩陣
    valid_embeddings = np.array([embeddings[i] for i in valid_indices])

    # SpectralClusterer 分群
    min_clusters = 2 if num_speakers is None else num_speakers
    max_clusters = 8 if num_speakers is None else num_speakers

    refinement_opts = refinement.RefinementOptions(
        gaussian_blur_sigma=1,
        p_percentile=0.95,
        thresholding_soft_multiplier=0.01,
        thresholding_type=refinement.ThresholdType.RowMax,
        symmetrize_type=refinement.SymmetrizeType.Max,
    )

    try:
        clusterer = SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            refinement_options=refinement_opts,
        )
        cluster_labels = clusterer.predict(valid_embeddings)
    except Exception as e:
        print(f"[diarize] 分群失敗: {e}，所有段落標記為 Speaker 1")
        return [0] * len(segments)

    # ── 餘弦相似度二次校正 ──
    unique_labels = sorted(set(cluster_labels))
    if len(unique_labels) > 1:
        centroids = {}
        for label in unique_labels:
            mask = [i for i, l in enumerate(cluster_labels) if l == label]
            centroids[label] = np.mean(valid_embeddings[mask], axis=0)
        reassigned = 0
        for idx in range(len(cluster_labels)):
            emb = valid_embeddings[idx]
            assigned = cluster_labels[idx]
            assigned_sim = float(np.dot(emb, centroids[assigned]))
            best_label, best_sim = assigned, assigned_sim
            for label, centroid in centroids.items():
                sim = float(np.dot(emb, centroid))
                if sim > best_sim:
                    best_label, best_sim = label, sim
            if best_label != assigned and (best_sim - assigned_sim) > 0.1:
                cluster_labels[idx] = best_label
                reassigned += 1
        if reassigned > 0:
            print(f"[diarize] 餘弦校正 {reassigned} 段")

    # 映射回所有段落
    speaker_labels = [None] * len(segments)
    for idx, valid_idx in enumerate(valid_indices):
        speaker_labels[valid_idx] = int(cluster_labels[idx])

    # 填補跳過的段落
    last_valid = 0
    for i in range(len(speaker_labels)):
        if speaker_labels[i] is not None:
            last_valid = speaker_labels[i]
        else:
            speaker_labels[i] = last_valid

    # 多數決平滑（窗口 5）
    changed = 0
    smoothed = list(speaker_labels)
    for i in range(len(smoothed)):
        start = max(0, i - 2)
        end = min(len(smoothed), i + 3)
        window = speaker_labels[start:end]
        majority = Counter(window).most_common(1)[0][0]
        if speaker_labels[i] != majority:
            smoothed[i] = majority
            changed += 1
    speaker_labels = smoothed
    if changed > 0:
        print(f"[diarize] 平滑修正 {changed} 段")

    # 按首次出現順序重新編號
    seen = {}
    renumber_map = {}
    counter = 0
    for label in speaker_labels:
        if label not in seen:
            seen[label] = True
            renumber_map[label] = counter
            counter += 1
    speaker_labels = [renumber_map[l] for l in speaker_labels]

    n_speakers = len(set(speaker_labels))
    print(f"[diarize] 完成（{n_speakers} 位講者）")

    return speaker_labels


# ── 模型載入 ──

def _get_model_faster(model_size: str):
    """faster-whisper 模型"""
    from faster_whisper import WhisperModel
    key = f"fw:{model_size}"
    if key not in _models:
        print(f"[載入模型] {model_size} (faster-whisper, device={_device}, compute={_compute_type})")
        _models[key] = WhisperModel(model_size, device=_device, compute_type=_compute_type)
        print(f"[模型就緒] {model_size}")
    return _models[key]


def _get_model_openai(model_size: str):
    """openai-whisper 模型"""
    import whisper as openai_whisper
    # openai-whisper 模型名稱對應：large-v3-turbo → turbo, large-v3 → large
    name_map = {
        "large-v3-turbo": "turbo",
        "large-v3": "large-v3",
        "medium.en": "medium.en",
        "small.en": "small.en",
        "base.en": "base.en",
    }
    ow_name = name_map.get(model_size, model_size)
    key = f"ow:{ow_name}"
    if key not in _models:
        print(f"[載入模型] {ow_name} (openai-whisper, device={_torch_device})")
        _models[key] = openai_whisper.load_model(ow_name, device=_torch_device)
        print(f"[模型就緒] {ow_name}")
    return _models[key], ow_name


# ── 辨識函式 ──

def _transcribe_faster(wav_path, model_size, language):
    """faster-whisper 辨識"""
    m = _get_model_faster(model_size)
    t0 = time.monotonic()
    segments_iter, info = m.transcribe(wav_path, language=language, beam_size=5, vad_filter=True)
    segments = []
    full_text = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
            full_text.append(text)
    return segments, full_text, round(info.duration, 1), round(time.monotonic() - t0, 1)


def _transcribe_faster_stream(wav_path, model_size, language):
    """faster-whisper 串流版，yield (segment_dict, duration) per segment"""
    m = _get_model_faster(model_size)
    segments_iter, info = m.transcribe(wav_path, language=language, beam_size=5, vad_filter=True)
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            yield {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text}, info.duration


class _ProgressCapture:
    """攔截 stdout，解析 openai-whisper verbose 輸出追蹤辨識進度。
    whisper verbose=True 每段輸出格式: [00:00.000 --> 00:30.000]  text..."""

    _TS_RE = re.compile(r'\[[\d:.]+\s*-->\s*([\d:.]+)\]')

    def __init__(self, original, progress_q, audio_duration):
        self._orig = original
        self._q = progress_q
        self._duration = audio_duration

    def write(self, text):
        self._orig.write(text)
        m = self._TS_RE.search(text)
        if m and self._duration > 0:
            secs = self._parse_ts(m.group(1))
            if secs is not None:
                pct = min(secs / self._duration, 1.0)
                self._q.put(("progress", secs, self._duration, pct))
        return len(text) if text else 0

    @staticmethod
    def _parse_ts(ts_str):
        parts = ts_str.split(':')
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
        return None

    def flush(self):
        self._orig.flush()


def _transcribe_openai(wav_path, model_size, language, progress_q=None):
    """openai-whisper 辨識。progress_q: Queue，用於回報辨識進度。"""
    m, ow_name = _get_model_openai(model_size)

    # 取得音訊時長
    audio_duration = 0
    if progress_q is not None:
        try:
            import whisper as _ow
            audio = _ow.load_audio(wav_path)
            audio_duration = len(audio) / 16000
            progress_q.put(("duration", audio_duration))
        except Exception:
            pass

    t0 = time.monotonic()

    # 有 progress_q 時用 verbose=True + stdout 攔截追蹤進度
    if progress_q is not None and audio_duration > 0:
        old_stdout = sys.stdout
        sys.stdout = _ProgressCapture(old_stdout, progress_q, audio_duration)
        try:
            result = m.transcribe(wav_path, language=language, beam_size=5, verbose=True)
        finally:
            sys.stdout = old_stdout
    else:
        result = m.transcribe(wav_path, language=language, beam_size=5)

    segments = []
    full_text = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:
            segments.append({"start": round(seg["start"], 3), "end": round(seg["end"], 3), "text": text})
            full_text.append(text)
    # openai-whisper 不直接回傳 duration，從最後一段取
    duration = round(segments[-1]["end"], 1) if segments else 0
    return segments, full_text, duration, round(time.monotonic() - t0, 1)


# ── API ──

@app.get("/health")
def health():
    """健康檢查"""
    return {
        "status": "ok",
        "gpu": _device == "cuda",
        "device": _device,
        "backend": _backend,
        "diarize": _HAS_DIARIZE,
    }


@app.get("/v1/status")
def status():
    """伺服器狀態：忙碌狀態 + 磁碟空間"""
    task = _get_active_task()
    busy = task is not None
    elapsed = round(time.time() - task["started"], 1) if busy else 0

    # /tmp 磁碟空間（暫存檔寫入處）
    disk = shutil.disk_usage(tempfile.gettempdir())
    disk_free_gb = round(disk.free / (1024 ** 3), 1)
    disk_total_gb = round(disk.total / (1024 ** 3), 1)

    result = {
        "busy": busy,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": disk_total_gb,
    }
    if busy:
        result["task"] = {
            "type": task["type"],
            "model": task["model"],
            "language": task["language"],
            "elapsed": elapsed,
            "client_ip": task["client_ip"],
        }
    return result


@app.get("/models")
def list_models():
    """列出已快取的模型"""
    cached = set()
    cached.update(k.split(":", 1)[1] for k in _models.keys())
    # 掃描 HuggingFace cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            name = repo.repo_id
            if name.startswith("Systran/faster-whisper-"):
                cached.add(name.replace("Systran/faster-whisper-", ""))
            elif name.startswith("guillaumekln/faster-whisper-"):
                cached.add(name.replace("guillaumekln/faster-whisper-", ""))
    except Exception:
        pass
    # openai-whisper 模型放在 ~/.cache/whisper/
    whisper_cache = os.path.expanduser("~/.cache/whisper")
    if os.path.isdir(whisper_cache):
        # 檔名格式: large-v3-turbo.pt, medium.en.pt 等
        for f in os.listdir(whisper_cache):
            if f.endswith(".pt"):
                cached.add(f[:-3])
    return {"models": sorted(cached)}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("large-v3-turbo"),
    language: str = Form("en"),
    stream: str = Form("false"),
):
    """接收音訊檔，回傳辨識結果（stream=true 時串流 NDJSON）"""
    client_ip = request.client.host if request.client else ""
    _set_active_task("transcribe", model, language, client_ip)

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # 串流模式（NDJSON）
        if stream.lower() == "true":
            tmp_path = tmp.name

            if _backend == "faster-whisper":
                def generate():
                    t0 = time.monotonic()
                    count = 0
                    dur = 0
                    cancelled = False
                    try:
                        for seg, dur in _transcribe_faster_stream(tmp_path, model, language):
                            count += 1
                            yield json.dumps({
                                "type": "segment", "index": count - 1,
                                "start": seg["start"], "end": seg["end"],
                                "text": seg["text"], "duration": round(dur, 1),
                            }) + "\n"
                        proc_time = round(time.monotonic() - t0, 1)
                        yield json.dumps({
                            "type": "done", "total_segments": count,
                            "duration": round(dur, 1), "processing_time": proc_time,
                            "device": _device,
                        }) + "\n"
                    except GeneratorExit:
                        cancelled = True
                        elapsed = round(time.monotonic() - t0, 1)
                        print(f"[取消] 客戶端中斷連線（{elapsed:.1f}s），faster-whisper 辨識已停止")
                        return
                    except Exception as e:
                        yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
                    finally:
                        _clear_active_task()
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            else:
                # openai-whisper：辨識中發心跳（含進度），完成後逐段回傳
                def generate():
                    import concurrent.futures
                    t0 = time.monotonic()
                    progress_q = queue.Queue()
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = pool.submit(_transcribe_openai, tmp_path, model, language,
                                         progress_q=progress_q)
                    cancelled = False
                    audio_dur = 0
                    last_pct = 0
                    last_pos = 0
                    try:
                        while not future.done():
                            # 讀取 progress queue 中的最新進度
                            while not progress_q.empty():
                                try:
                                    msg = progress_q.get_nowait()
                                    if msg[0] == "duration":
                                        audio_dur = msg[1]
                                    elif msg[0] == "progress":
                                        last_pos = msg[1]
                                        last_pct = msg[3]
                                except queue.Empty:
                                    break
                            elapsed = round(time.monotonic() - t0, 1)
                            hb = {"type": "heartbeat", "elapsed": elapsed}
                            if audio_dur > 0:
                                hb["progress"] = round(last_pct, 3)
                                hb["current"] = round(last_pos, 1)
                                hb["duration"] = round(audio_dur, 1)
                            yield json.dumps(hb) + "\n"
                            time.sleep(2)
                        segments, full_text, duration, proc_time = future.result()
                        for i, seg in enumerate(segments):
                            yield json.dumps({
                                "type": "segment", "index": i,
                                "start": seg["start"], "end": seg["end"],
                                "text": seg["text"], "duration": round(duration, 1),
                            }) + "\n"
                        yield json.dumps({
                            "type": "done", "total_segments": len(segments),
                            "duration": round(duration, 1), "processing_time": proc_time,
                            "device": _device,
                        }) + "\n"
                    except GeneratorExit:
                        cancelled = True
                        future.cancel()
                        elapsed = round(time.monotonic() - t0, 1)
                        print(f"[取消] 客戶端中斷連線（{elapsed:.1f}s），等待 openai-whisper 辨識執行緒結束...")
                        # 等 transcribe thread 真正結束再清理（GPU 仍在跑）
                        pool.shutdown(wait=True)
                        print(f"[取消] openai-whisper 執行緒已結束")
                        return
                    except Exception as e:
                        yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
                    finally:
                        if not cancelled:
                            pool.shutdown(wait=False)
                        _clear_active_task()
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            # 串流模式由 generator 負責刪除暫存檔，不走 finally
            return StreamingResponse(generate(), media_type="text/x-ndjson")

        # 非串流模式（用 asyncio.to_thread 避免阻塞 event loop）
        try:
            if _backend == "openai-whisper":
                segments, full_text, duration, proc_time = await asyncio.to_thread(
                    _transcribe_openai, tmp.name, model, language)
            else:
                segments, full_text, duration, proc_time = await asyncio.to_thread(
                    _transcribe_faster, tmp.name, model, language)
        except Exception as e:
            print(f"[錯誤] 辨識失敗: {model} — {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"辨識失敗: {model}", "detail": str(e)},
            )

        return {
            "text": " ".join(full_text),
            "segments": segments,
            "language": language,
            "model": model,
            "duration": duration,
            "processing_time": proc_time,
            "device": _device,
            "backend": _backend,
        }
    finally:
        # 非串流模式清理（串流模式由 generator 清理，不分後端）
        if stream.lower() != "true":
            _clear_active_task()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@app.post("/v1/audio/diarize")
async def diarize(
    request: Request,
    file: UploadFile = File(...),
    segments: str = Form(...),
    num_speakers: int = Form(0),
):
    """接收音訊檔 + segments JSON，回傳講者辨識結果"""
    from fastapi.responses import JSONResponse

    if not _HAS_DIARIZE:
        return JSONResponse(
            status_code=500,
            content={"error": "resemblyzer/spectralcluster 未安裝，無法執行講者辨識"},
        )

    # 解析 segments JSON
    try:
        seg_list = json.loads(segments)
        if not isinstance(seg_list, list):
            raise ValueError("segments 必須是 list")
        for s in seg_list:
            if not all(k in s for k in ("start", "end", "text")):
                raise ValueError("每個 segment 必須含 start, end, text")
    except (json.JSONDecodeError, ValueError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"segments JSON 格式錯誤: {e}"},
        )

    client_ip = request.client.host if request.client else ""
    _set_active_task("diarize", "resemblyzer", language="", client_ip=client_ip)

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        ns = num_speakers if num_speakers > 0 else None
        t0 = time.monotonic()

        try:
            speaker_labels = await asyncio.to_thread(_diarize, tmp.name, seg_list, num_speakers=ns)
        except Exception as e:
            print(f"[錯誤] diarize 失敗: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"講者辨識失敗: {e}"},
            )

        proc_time = round(time.monotonic() - t0, 2)

        if speaker_labels is None:
            # 無法提取聲紋，降級全部 Speaker 0
            speaker_labels = [0] * len(seg_list)

        return {
            "speaker_labels": speaker_labels,
            "num_speakers": len(set(speaker_labels)),
            "processing_time": proc_time,
            "device": _torch_device,
        }
    finally:
        _clear_active_task()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jt-whisper-server")
    parser.add_argument("--port", type=int, default=8978)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"[jt-whisper-server] 啟動 {args.host}:{args.port} (backend={_backend}, device={_device})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
