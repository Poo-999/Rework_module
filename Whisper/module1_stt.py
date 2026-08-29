"""
Module 1: Speech-to-Text
========================
แยกข้อความคำตอบจากเสียงในวิดีโอสัมภาษณ์ โดยใช้ faster-whisper
- รองรับภาษาไทย (ใช้ model size "large-v3" เพื่อความแม่นยำสูงสุด)
- คืนค่า transcript แบบเต็ม + timestamp ระดับ segment (ใช้ sync กับ audio/facial module ทีหลัง)

ติดตั้ง dependencies:
    pip install faster-whisper "moviepy>=2.0" --break-system-packages

หมายเหตุ:
- faster-whisper ต้องใช้ ffmpeg ในระบบ (apt install ffmpeg)
- ใช้ moviepy >= 2.0 เท่านั้น เพราะ import path เปลี่ยนจาก `moviepy.editor`
  เป็น root package ตรง ๆ (`from moviepy import VideoFileClip`) ตั้งแต่ moviepy 2.x
  ถ้าใช้ moviepy 1.x ต้องเปลี่ยน import กลับเป็น `from moviepy.editor import VideoFileClip`
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List
from tabulate import tabulate

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
from moviepy import VideoFileClip

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
        output = {
            "video_path": input_path,
            "audio_path": audio_path,
            "full_transcript": full_text,
            "segments": [asdict(s) for s in segments],
        }

        if save_json:
            out_path = os.path.splitext(input_path)[0] + "_transcript.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"บันทึกผลลัพธ์ที่: {out_path}")

        # เรียกใช้ calculate_rtf() แทนคำนวณซ้ำ (ฟังก์ชันนี้อยู่ท้ายไฟล์เดียวกัน
        # เรียกได้ปกติเพราะ Python จะมองหาฟังก์ชันตอนถูกเรียกจริง ไม่ใช่ตอนนิยาม class)
        logger.info(calculate_rtf(time.time() - t0, duration))

        logger.info(f"Pipeline เสร็จสมบูรณ์ ใช้เวลารวม {time.time() - t0:.1f} วินาที")
        return output

# ---------- Metric สำหรับ evaluate module นี้ ----------
def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate: เทียบ transcript ที่ได้จาก whisper กับ manual transcript (ground truth)
    ใช้ pythainlp ตัดคำก่อน เพราะภาษาไทยไม่มี word boundary
    pip install jiwer pythainlp --break-system-packages
    """
    from pythainlp.tokenize import word_tokenize
    from jiwer import wer

    # กรอง whitespace-only token ออก (newmm เก็บช่องว่างเป็น token แยก
    # ถ้าไม่กรอง jiwer จะนับจำนวนคำไม่ตรงกับ list เดิม ทำให้ WER คลาดเคลื่อน)
    ref_tokens = [t for t in word_tokenize(reference, engine="newmm") if t.strip() != ""]
    hyp_tokens = [t for t in word_tokenize(hypothesis, engine="newmm") if t.strip() != ""]
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)
    return f"Word Error Rate (WER): {wer(ref_str, hyp_str) * 100:.2f}%"


def calculate_cer(reference: str, hypothesis: str) -> str:
    """
    Character Error Rate: เทียบ transcript ที่ได้จาก whisper กับ manual transcript (ground truth)
    คำนวณระดับตัวอักษร ไม่ต้อง tokenize คำก่อน (ส่ง string ดิบเข้าไปได้เลย)
    ข้อดีกว่า WER: ไม่ผูกกับ tokenizer เลย จึงไม่มีปัญหาเรื่องตัดคำผิด/ตัดคำไม่ตรงกัน
    ที่เจอกับ WER (เช่นกรณี "กวีพันธ์" ถูกตัดเป็น "กวี"+"พันธ์") เหมาะใช้เป็น metric
    เสริมคู่กับ WER เพื่อดูภาพให้ครบทั้ง 2 มุม
    pip install jiwer --break-system-packages
    """
    from jiwer import cer

    return f"Character Error Rate (CER): {cer(reference, hypothesis) * 100:.2f}%"


def calculate_rtf(processing_time: float, duration: float) -> str:
    """
    Real-Time Factor: เวลาที่ใช้ประมวลผล ÷ ความยาวเสียงจริง
    ต่ำกว่า 1 = เร็วกว่า real-time (ดี), มากกว่า 1 = ช้ากว่า real-time
    ไม่ต้องคูณ 100 หรือใส่ % เหมือน WER/CER เพราะ RTF เป็นค่าเทียบสัดส่วนตรง ๆ
    กับเลข 1 ไม่ใช่ percentage
    """
    return f"Real-Time Factor (RTF): {processing_time / duration if duration > 0 else float('inf'):.2f}"

#---------- Metric สำหรับ evaluate module นี้ (ต่อ) ----------
import csv
import os
from pythainlp.tokenize import word_tokenize
import jiwer

# NOTE: เคยมี custom dictionary ตรงนี้ (กัน "กวีพันธ์" ถูกตัดเป็น "กวี"+"พันธ์")
# ตัดออกไปก่อนชั่วคราว เพื่อให้ export_wer_to_csv() ใช้ tokenizer แบบเดียวกับ
# calculate_wer() เป๊ะ ๆ จะได้เทียบผล "มี initial_prompt vs ไม่มี" ได้ตรงประเด็น
# โดยไม่มีตัวแปรเรื่อง tokenizer มาปน — ไว้ค่อยเพิ่มกลับเข้ามาทีหลัง


def _tokenize_th(text: str) -> list:
    """ตัดคำไทยด้วย word_tokenize มาตรฐาน (ไม่มี custom dictionary)
    กรอง whitespace-only token ออกด้วย (ดูเหตุผลใน docstring ของ export_wer_to_csv)"""
    return [t for t in word_tokenize(text, engine="newmm") if t.strip() != ""]


def export_wer_to_csv(
    reference_text: str,
    hypothesis_text: str,
    output_prefix: str = "stt_eval",
    only_errors: bool = True,
):
    """
    วิเคราะห์และบันทึกผลการประเมิน WER และ Word Alignment เป็นไฟล์ CSV

    :param reference_text: ข้อความเฉลย (Ground Truth)
    :param hypothesis_text: ข้อความที่ Whisper ถอดเสียงได้
    :param output_prefix: ชื่อขึ้นต้นของไฟล์ CSV (เช่น 'stt_eval')
    :param only_errors: บันทึกเฉพาะจุดที่ผิดพลาดหรือไม่ (True = เฉพาะที่ผิด, False = บันทึกทุกคำ)
    """
    # 1. ตัดคำภาษาไทย (ใช้ tokenizer เดียวกับ calculate_wer() เพื่อให้ผลตรงกัน)
    # กรอง whitespace-only token ออก — สำคัญมาก: newmm เก็บช่องว่างระหว่างคำเป็น
    # token แยกต่างหาก (เช่น ' ') ถ้าไม่กรองออก ตอน " ".join(tokens) จะได้ช่องว่างซ้ำซ้อน
    # แล้ว jiwer จะ collapse ช่องว่างซ้ำเอง ทำให้จำนวนคำที่ jiwer นับ "ไม่ตรง" กับ
    # len(ref_tokens)/len(hyp_tokens) ที่แท้จริง ผลคือ chunk.ref_start_idx / ref_end_idx
    # ที่ jiwer คืนมาจะไม่ตรงตำแหน่งกับ ref_tokens list ทำให้คำใน CSV เพี้ยน/สลับตำแหน่ง
    # และ WER สูงเกินจริง (นี่คือสาเหตุที่คำใน alignment CSV ก่อนหน้านี้ดูเพี้ยน)
    ref_tokens = _tokenize_th(reference_text)
    hyp_tokens = _tokenize_th(hypothesis_text)

    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)

    # 2. ประมวลผล Alignment ด้วย jiwer
    out = jiwer.process_words(ref_str, hyp_str)

    # ---------------------------------------------------------
    # 3. บันทึกไฟล์ที่ 1: สรุปภาพรวม (Summary CSV)
    # ---------------------------------------------------------
    summary_filename = f"{output_prefix}_summary.csv"
    summary_rows = [
        ["Metric", "Value", "Description"],
        ["Total Reference Words (N)", len(ref_tokens), "จำนวนคำทั้งหมดในเฉลย"],
        ["Hits / Correct (C)", out.hits, "คำที่ถอดเสียงได้ถูกต้อง"],
        ["Substitutions (S)", out.substitutions, "คำที่สะกดผิดหรือแทนที่ด้วยคำอื่น"],
        ["Deletions (D)", out.deletions, "คำที่ตกหล่นไป"],
        ["Insertions (I)", out.insertions, "คำที่โมเดลใส่เกินมา"],
        ["Word Error Rate (WER %)", f"{out.wer * 100:.2f}%", "อัตราความผิดพลาดระดับคำ"],
    ]

    with open(summary_filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)

    print(f"✓ บันทึกสรุปผลลัพธ์ภาพรวมเรียบร้อย: {summary_filename}")

    # ---------------------------------------------------------
    # 4. บันทึกไฟล์ที่ 2: รายการ Alignment ทีละจุด (Alignment CSV)
    # ---------------------------------------------------------
    alignment_filename = f"{output_prefix}_alignment.csv"
    alignment_rows = [
        ["No", "Error_Type", "Error_Type_TH", "Reference_Word", "Hypothesis_Word", "Status"]
    ]

    row_index = 1
    for chunk in out.alignments[0]:
        op_type = chunk.type
        ref_chunk = " ".join(ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx])
        hyp_chunk = " ".join(hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx])

        # ถ้าเลือก only_errors=True จะข้ามคำที่ถูกต้องไป
        if only_errors and op_type == "equal":
            continue

        type_mapping = {
            "equal": ("Correct", "ถูกต้อง", "ตรงกับเฉลย"),
            "substitute": ("Substitution", "คำสะกดผิด / ถูกแทนที่", "ผิดคำ"),
            "delete": ("Deletion", "คำตกหล่น", "หายไป"),
            "insert": ("Insertion", "คำเกินมา", "เกินมา"),
        }

        en_type, th_type, status = type_mapping.get(
            op_type, (op_type, op_type, "ไม่ระบุ")
        )

        alignment_rows.append(
            [
                row_index,
                en_type,
                th_type,
                ref_chunk if ref_chunk else "-",
                hyp_chunk if hyp_chunk else "-",
                status,
            ]
        )
        row_index += 1

    with open(alignment_filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(alignment_rows)

    print(f"✓ บันทึกรายการ Alignment ละเอียดเรียบร้อย: {alignment_filename}")



# ---------- ตัวอย่างการใช้งาน ----------
if __name__ == "__main__":
    # GPU VRAM < 6GB: ใช้ "medium" หรือ "small" + compute_type="int8_float16"
    stt = SpeechToTextModule(model_size="medium", device="cuda", compute_type="int8_float16")

    # process_video() รองรับทั้งไฟล์วิดีโอและไฟล์เสียง (.wav) ที่แตกไว้แล้ว
    # ถ้าเป็น .wav อยู่แล้วจะข้ามขั้นตอนแตกเสียงจากวิดีโอ (moviepy) โดยอัตโนมัติ
    result = stt.process_video(r"D:\Modeling\Rework_model\Whisper\reference1.m4a",initial_prompt="จักริน กวีพันธ์, ซอฟต์แวร์, โปรเจกต์")
    print(result["full_transcript"])

    print(calculate_wer(reference="""สวัสดีครับ ผมชื่อ จักริน กวีพันธ์ มีความสนใจในตำแหน่งนักพัฒนาซอฟต์แวร์ของบริษัทนี้ครับ ที่ผ่านมาผมมีประสบการณ์ในการพัฒนาโปรเจกต์ทั้งฝั่งหน้าบ้านและหลังบ้าน ได้ทำงานร่วมกับทีมในการออกแบบระบบและแก้ไขปัญหาตามความต้องการของผู้ใช้งาน ผมเป็นคนที่ชอบเรียนรู้เทคโนโลยีใหม่ ๆ อยู่เสมอ และสามารถปรับตัวเข้ากับสภาพแวดล้อมการทำงานได้อย่างรวดเร็วครับ""", hypothesis=result["full_transcript"]))
    print(calculate_cer(reference="""สวัสดีครับ ผมชื่อ จักริน กวีพันธ์ มีความสนใจในตำแหน่งนักพัฒนาซอฟต์แวร์ของบริษัทนี้ครับ ที่ผ่านมาผมมีประสบการณ์ในการพัฒนาโปรเจกต์ทั้งฝั่งหน้าบ้านและหลังบ้าน ได้ทำงานร่วมกับทีมในการออกแบบระบบและแก้ไขปัญหาตามความต้องการของผู้ใช้งาน ผมเป็นคนที่ชอบเรียนรู้เทคโนโลยีใหม่ ๆ อยู่เสมอ และสามารถปรับตัวเข้ากับสภาพแวดล้อมการทำงานได้อย่างรวดเร็วครับ""", hypothesis=result["full_transcript"]))
    export_wer_to_csv(
        reference_text="""สวัสดีครับ ผมชื่อ จักริน กวีพันธ์ มีความสนใจในตำแหน่งนักพัฒนาซอฟต์แวร์ของบริษัทนี้ครับ ที่ผ่านมาผมมีประสบการณ์ในการพัฒนาโปรเจกต์ทั้งฝั่งหน้าบ้านและหลังบ้าน ได้ทำงานร่วมกับทีมในการออกแบบระบบและแก้ไขปัญหาตามความต้องการของผู้ใช้งาน ผมเป็นคนที่ชอบเรียนรู้เทคโนโลยีใหม่ ๆ อยู่เสมอ และสามารถปรับตัวเข้ากับสภาพแวดล้อมการทำงานได้อย่างรวดเร็วครับ""",
        hypothesis_text=result["full_transcript"],
    )