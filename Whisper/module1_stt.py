"""
Module 1: Speech-to-Text
========================
แยกข้อความคำตอบจากเสียงในวิดีโอสัมภาษณ์ โดยใช้ faster-whisper
- รองรับภาษาไทย (ใช้ model size "large-v3" เพื่อความแม่นยำสูงสุด)
- คืนค่า transcript แบบเต็ม + timestamp ระดับ segment (ใช้ sync กับ audio/facial module ทีหลัง)

ติดตั้ง dependencies:
    pip install faster-whisper "moviepy<2.0" --break-system-packages

หมายเหตุ:
- faster-whisper ต้องใช้ ffmpeg ในระบบ (apt install ffmpeg)
- ใช้ moviepy 1.x (import ผ่าน `from moviepy.editor import VideoFileClip`)
  ถ้าจะอัปเกรดเป็น moviepy 2.x ในอนาคต ต้องเปลี่ยน import เป็น
  `from moviepy import VideoFileClip` (root package) แทน
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List

def _add_nvidia_dll_dirs():
    """
    Windows only: faster-whisper (ผ่าน ctranslate2) ต้องการ cublas64_12.dll และ cudnn64_9.dll
    ถ้าติดตั้งผ่าน pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) ไฟล์ .dll จะอยู่ใน
    site-packages\\nvidia\\<package>\\bin\\ แต่ Windows จะหาไม่เจอเองถ้าไม่เพิ่ม path นี้
    เข้า DLL search path ก่อน (os.add_dll_directory ใช้ได้ตั้งแต่ Python 3.8+)
    ถ้ายัง error "Library ... is not found" หลังรันฟังก์ชันนี้ ให้ตรวจว่า
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 สำเร็จจริงในเดียวกับ venv ที่รันอยู่
    """
    if sys.platform != "win32":
        return

    candidate_dirs = set()
    try:
        import site
        candidate_dirs.update(site.getsitepackages())
    except Exception:
        pass
    candidate_dirs.add(os.path.join(sys.prefix, "Lib", "site-packages"))

    added = []
    for base in candidate_dirs:
        nvidia_dir = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg_name in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg_name, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
                # เสริม: เติมเข้า PATH โดยตรงด้วย เพราะบางกรณี native DLL loading
                # ของ ctranslate2 (implicit dependency loading) พึ่ง PATH มากกว่า
                # os.add_dll_directory() เพียงอย่างเดียว
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                added.append(bin_dir)
                # แสดงรายชื่อ .dll ทั้งหมดที่เจอในโฟลเดอร์นี้ เพื่อ debug ว่ามี dll
                # ที่ยัง missing อยู่หรือเปล่า (เช่น cudart64_12.dll ที่บาง dll ต้องพึ่งต่อ)
                dlls = [f for f in os.listdir(bin_dir) if f.lower().endswith(".dll")]
                print(f"[dll setup]   พบใน {bin_dir}: {dlls}")

    if added:
        print(f"[dll setup] เพิ่ม DLL search path แล้ว ({len(added)} โฟลเดอร์): {added}")
    else:
        print("[dll setup] ไม่พบโฟลเดอร์ nvidia ใน site-packages "
              "(ถ้าเจอ error cublas อีก ให้เช็คว่า pip install nvidia-cublas-cu12 "
              "nvidia-cudnn-cu12 ลงใน venv เดียวกับที่รันสคริปต์นี้จริงหรือไม่)")


_add_nvidia_dll_dirs()

# เปิด progress bar ของ huggingface_hub ตอนดาวน์โหลดโมเดล (ถ้ายังไม่มีไฟล์ในเครื่อง)
# ต้องตั้งก่อน import faster_whisper เพื่อให้มีผล
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stt_module")


@dataclass
class TranscriptSegment:
    id: int
    start: float          # วินาที
    end: float             # วินาที
    text: str
    avg_logprob: float     # ความมั่นใจของโมเดล (ยิ่งใกล้ 0 ยิ่งมั่นใจ)
    no_speech_prob: float  # โอกาสที่ช่วงนี้ไม่มีเสียงพูด (ใช้กรอง noise)


class SpeechToTextModule:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        """
        model_size: 'small' สำหรับทดสอบเร็ว ๆ / 'large-v3' สำหรับ production
        device: 'cuda' ถ้ามี GPU, ไม่งั้นใช้ 'cpu'
        compute_type: 'float16' (GPU) หรือ 'int8' (CPU เร็วขึ้นแต่แม่นยำลดลงเล็กน้อย)
        """
        logger.info(f"กำลังโหลดโมเดล Whisper '{model_size}' (device={device}, compute_type={compute_type})...")
        logger.info("ถ้าเป็นการรันครั้งแรก ระบบจะดาวน์โหลดโมเดลจาก HuggingFace ก่อน (large-v3 ~3GB) อาจใช้เวลาสักครู่")
        t0 = time.time()
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"โหลดโมเดลเสร็จแล้ว ใช้เวลา {time.time() - t0:.1f} วินาที")

    def extract_audio(self, video_path: str, output_audio_path: str = None) -> str:
        """แยกไฟล์เสียงออกจากวิดีโอ เป็น .wav 16kHz mono (format ที่ whisper ต้องการ)"""
        if output_audio_path is None:
            output_audio_path = os.path.splitext(video_path)[0] + "_audio.wav"

        logger.info(f"กำลังแตกไฟล์เสียงจาก: {video_path}")
        t0 = time.time()
        clip = VideoFileClip(video_path)
        logger.info(f"เปิดวิดีโอสำเร็จ ความยาว {clip.duration:.1f} วินาที กำลัง export เสียง...")
        clip.audio.write_audiofile(
            output_audio_path,
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],  # mono channel
            logger="bar",  # โชว์ progress bar ตอน export (เดิมปิดด้วย logger=None เลยดูเหมือนค้าง)
        )
        clip.close()
        logger.info(f"แตกไฟล์เสียงเสร็จแล้ว ({time.time() - t0:.1f}s) -> {output_audio_path}")
        return output_audio_path

    def transcribe(self, audio_path: str, language: str = "th", initial_prompt: str = None) -> List[TranscriptSegment]:
        """
        ถอดเสียงเป็นข้อความ พร้อม timestamp ต่อ segment

        initial_prompt: ข้อความ "บอกใบ้" ให้ Whisper รู้จักคำเฉพาะก่อนถอดเสียง
                        เหมาะกับชื่อผู้สมัคร/ศัพท์เทคนิคที่ดึงมาจาก resume
                        เช่น "จักริน กวีพันธ์, ซอฟต์แวร์, โปรเจกต์"
                        ปล่อยเป็น None ได้ถ้าไม่มีข้อมูลตรงนี้
        """
        logger.info(f"กำลังถอดเสียงจาก: {audio_path} (ภาษา={language})")
        if initial_prompt:
            logger.info(f"ใช้ initial_prompt ช่วย bias การถอดเสียง: {initial_prompt!r}")
        t0 = time.time()
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,          # กรองช่วงเงียบออกอัตโนมัติ
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,     # เปิดเป็น True ถ้าต้องการ timestamp ระดับคำ (ใช้เพิ่ม latency)
        )
        logger.info(f"ความยาวเสียงที่ตรวจพบ: {info.duration:.1f}s | เริ่มถอดเสียงทีละ segment...")

        # NOTE: model.transcribe() คืนค่า segments เป็น generator (lazy)
        # ตัวโมเดลจะยังไม่ทำงานจริงจนกว่าจะ iterate ผ่าน for-loop นี้
        # เดิมโค้ดไม่ print อะไรเลยตรงนี้ เลยดูเหมือนค้าง ทั้งที่จริงกำลังถอดเสียงอยู่
        result = []
        for i, seg in enumerate(segments):
            elapsed = time.time() - t0
            logger.info(f"  [segment {i}] {seg.start:.1f}s-{seg.end:.1f}s "
                        f"(ผ่านไปแล้ว {elapsed:.1f}s): {seg.text.strip()[:60]}")
            result.append(
                TranscriptSegment(
                    id=i,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg.text.strip(),
                    avg_logprob=round(seg.avg_logprob, 3),
                    no_speech_prob=round(seg.no_speech_prob, 3),
                )
            )
        logger.info(f"ถอดเสียงเสร็จสิ้น รวม {len(result)} segments ใช้เวลา {time.time() - t0:.1f}s")
        return result, info.duration

    def process_video(self, input_path: str, save_json: bool = True, language: str = "th",
                       initial_prompt: str = None) -> dict:
        """
        Pipeline เต็ม: video/audio -> (แตกเสียงถ้าจำเป็น) -> transcript

        รองรับทั้งไฟล์วิดีโอ (.mp4, .mov, .avi, ...) และไฟล์เสียงที่แตกไว้แล้ว
        (.wav, .mp3, .m4a, ...) ถ้าเป็นไฟล์เสียงอยู่แล้วจะข้ามขั้นตอน extract_audio
        (ไม่ต้องเรียก moviepy) ทำให้เร็วขึ้นและลดจุดที่อาจพังจาก moviepy/ffmpeg

        initial_prompt: ส่งต่อให้ transcribe() — ใส่ชื่อผู้สมัคร/ศัพท์เทคนิคจาก resume
                        เพื่อช่วยให้ Whisper ถอดเสียงคำเฉพาะเหล่านี้แม่นขึ้น
        """
        t0 = time.time()

        AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
        ext = os.path.splitext(input_path)[1].lower()

        if ext in AUDIO_EXTENSIONS:
            logger.info(f"ไฟล์นำเข้าเป็นไฟล์เสียงอยู่แล้ว ({ext}) ข้ามขั้นตอนแตกไฟล์เสียงจากวิดีโอ")
            audio_path = input_path
        else:
            audio_path = self.extract_audio(input_path)

        segments, duration = self.transcribe(audio_path, language=language, initial_prompt=initial_prompt)

        full_text = " ".join(s.text for s in segments)
        processing_time_sec = round(time.time() - t0, 2)
        output = {
            "video_path": input_path,
            "audio_path": audio_path,
            "full_transcript": full_text,
            "segments": [asdict(s) for s in segments],
            # เก็บไว้ให้ metric module แยกใช้คำนวณ RTF ต่อ โดยไม่ต้องเปิดไฟล์เสียงซ้ำ
            "audio_duration_sec": round(duration, 2),
            "processing_time_sec": processing_time_sec,
        }

        if save_json:
            out_path = os.path.splitext(input_path)[0] + "_transcript.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"บันทึกผลลัพธ์ที่: {out_path}")

        logger.info(f"Pipeline เสร็จสมบูรณ์ ใช้เวลารวม {time.time() - t0:.1f} วินาที")
        return output





# ---------- ตัวอย่างการใช้งาน ----------
if __name__ == "__main__":
    # GPU VRAM < 6GB: ใช้ "medium" หรือ "small" + compute_type="int8_float16"
    path = r"D:\Modeling\Rework_model\models\whisper-th-medium-ct2"
    stt = SpeechToTextModule(model_size="medium", device="cuda", compute_type="int8_float16")

    # process_video() รองรับทั้งไฟล์วิดีโอและไฟล์เสียง (.wav) ที่แตกไว้แล้ว
    # ถ้าเป็น .wav อยู่แล้วจะข้ามขั้นตอนแตกเสียงจากวิดีโอ (moviepy) โดยอัตโนมัติ

    # mock initial prompt: ใส่ชื่อผู้สมัคร/ศัพท์เทคนิคจาก resume เพื่อ bias โมเดลให้ถอดเสียงคำเฉพาะเหล่านี้แม่นขึ้น
    mock_resume_data = {
    "name": "จักริน กวีพันธ์",
    "position": "Data Engineer",
    "skills": [
        "Data Pipeline",
        "Machine Learning",
        "Python",
        "SQL",
        "Hyperparameter Tuning",
        "Unit Test",
        "Data Cleaning",
    ],
    }
    def build_initial_prompt(resume_data: dict) -> str:
        """
        ประกอบ initial_prompt จากข้อมูล resume
        รวม ชื่อ + ตำแหน่งงาน + skill ทั้งหมด เป็นข้อความเดียว คั่นด้วย comma
        ตามรูปแบบที่ Whisper คาดหวัง (ประโยคสั้น ๆ ที่มีคำศัพท์ที่อยากให้รู้จัก)
        """
        parts = [resume_data["name"], resume_data["position"]] + resume_data["skills"]
        return ", ".join(parts)

    initial_prompt = build_initial_prompt(mock_resume_data)




    result = stt.process_video(r"D:\Modeling\Data\บทสัมภาษณ์\Interview.m4a", initial_prompt=initial_prompt)
    print(result["full_transcript"])
    