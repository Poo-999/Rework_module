"""
Module 2: Evaluation Metrics (WER / CER / RTF)
===============================================
ประเมินคุณภาพผลลัพธ์จาก module1_stt.py โดยเทียบกับ reference transcript
ที่พิมพ์ไว้เอง (ground truth) เทียบกันที่ระดับ "ทั้งไฟล์" (full transcript)
ไม่ใช่ต่อ segment เพราะ segment boundary ของ whisper ไม่ตรงกับที่คนพิมพ์ตามธรรมชาติ
ทำให้เฉลี่ยแล้วตัวเลขบิดเบือน

Metric ที่คำนวณ:
- CER (Character Error Rate) — ★ ใช้เป็นตัวชี้วัดหลัก ★
    เทียบระดับตัวอักษรตรง ๆ ไม่ต้องตัดคำ จึงไม่มีปัญหาเรื่อง tokenizer ตัดคำผิด
    เหมาะกับภาษาไทยที่ไม่มีช่องว่างคั่นคำ และงานที่มีคำทับศัพท์/ศัพท์เทคนิคเยอะ
    (เช่น "Data Engineer" / "ดาต้า เอนจิเนียร์") ซึ่งเป็นจุดอ่อนของ WER แบบตัดคำ
- WER (Word Error Rate) — ใช้เป็นข้อมูลประกอบเท่านั้น
    ตัดคำไทยด้วย pythainlp ก่อนคำนวณ แต่ pythainlp มักตัดคำทับศัพท์ผิดเพี้ยน
    (ตัด "ดาต้า" เป็น "ดา"+"ต้า") ทำให้ WER พุ่งสูงเกินจริงในงานที่มีศัพท์อังกฤษปนเยอะ
    จึงไม่ควรใช้ WER เป็นตัวตัดสินหลัก เก็บไว้ดูเป็นเบาะแสประกอบการอ่าน diff report
- RTF (Real-Time Factor): processing_time_sec / audio_duration_sec (ดึงจาก JSON ที่
                          module1_stt.py บันทึกไว้แล้ว ไม่ต้องเปิดไฟล์เสียง/วิดีโอซ้ำ)

นอกจากตัวเลข ยังสร้าง "diff report" (alignment) ที่ระบุว่าคำ/ตัวอักษรไหนถูกนับเป็น
substitute (S) / delete (D) / insert (I) เพื่อให้อธิบายได้ว่าทำไมถึงได้ค่า WER/CER นั้นมา

ติดตั้ง dependencies:
    pip install jiwer pythainlp --break-system-packages

วิธีใช้ (command line):
    python module2_metrics.py path/to/xxx_transcript.json path/to/reference.txt
    python module2_metrics.py path/to/xxx_transcript.json path/to/reference.txt --out result.json

    python module1_metrics.py "D:\Modeling\Data\Reviewer\Interview_transcript.json" "D:\Modeling\Data\Reviewer\interview.txt" --out result.json

หรือเรียกจาก python อื่นก็ได้ผ่านฟังก์ชัน evaluate()
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import jiwer
from pythainlp.tokenize import word_tokenize


# ---------- Text normalization ----------

def normalize_text(text: str) -> str:
    """
    ทำความสะอาดข้อความก่อนเทียบ:
    - ยุบช่องว่าง/ขึ้นบรรทัดใหม่ให้เหลือช่องว่างเดียว
    - ตัดช่องว่างหัวท้าย
    หมายเหตุ: ตั้งใจ "ไม่" ตัดเครื่องหมายวรรคตอนออกให้อัตโนมัติ เพราะถ้า reference
    กับ hypothesis ใช้เครื่องหมายวรรคตอนไม่ตรงกัน (เช่น whisper ใส่จุด/จุลภาคเอง)
    จะถูกนับเป็นจุดผิดที่ไม่เกี่ยวกับความถูกต้องของเนื้อหาจริง ๆ
    ถ้าพบว่า reference/hypothesis มีวรรคตอนไม่สอดคล้องกันเยอะ ให้พิจารณาเพิ่ม
    ขั้นตอนลบวรรคตอนออกทั้งคู่ก่อนเรียก evaluate()
    """
    return re.sub(r"\s+", " ", text.strip())


def thai_tokenize_for_wer(text: str) -> str:
    """
    ตัดคำไทยด้วย pythainlp แล้ว join กลับด้วยช่องว่าง เพื่อให้ jiwer (ซึ่ง split
    ด้วย whitespace เป็น default) นับ WER ระดับคำได้ถูกต้อง แทนที่จะ split
    ด้วยช่องว่างของข้อความไทยดิบ ๆ ซึ่งจะได้ "คำ" ที่จริงเป็นทั้งวลี
    """
    text = normalize_text(text)
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    tokens = [t for t in tokens if t.strip()]
    return " ".join(tokens)


# ---------- Diff / alignment report ----------

def build_diff_report(wer_output) -> str:
    """
    แปลง alignment ของ jiwer ให้เป็นรายงานที่อ่านง่าย ระบุว่าคำไหนถูกนับเป็น
    S (substitute) / D (delete) / I (insert) เทียบกับ reference
    """
    lines = []
    for ref_words, hyp_words, alignment in zip(
        wer_output.references, wer_output.hypotheses, wer_output.alignments
    ):
        has_error = False
        for chunk in alignment:
            ref_seg = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            hyp_seg = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            if chunk.type == "equal":
                continue
            has_error = True
            if chunk.type == "substitute":
                lines.append(f"  [S] แก้คำ: '{ref_seg}'  ->  '{hyp_seg}'")
            elif chunk.type == "delete":
                lines.append(f"  [D] หายไป (มีใน reference แต่ whisper ไม่ถอด): '{ref_seg}'")
            elif chunk.type == "insert":
                lines.append(f"  [I] เกินมา (whisper ถอดเพิ่มมาโดยไม่มีใน reference): '{hyp_seg}'")
        if not has_error:
            lines.append("  (ไม่มีจุดผิด ตรงกับ reference ทั้งหมด)")
    return "\n".join(lines)


# ---------- Core evaluation ----------

@dataclass
class MetricResult:
    wer: float
    cer: float
    rtf: Optional[float]
    wer_substitutions: int
    wer_deletions: int
    wer_insertions: int
    wer_hits: int
    cer_substitutions: int
    cer_deletions: int
    cer_insertions: int
    cer_hits: int
    diff_report: str


def compute_wer_cer(reference: str, hypothesis: str):
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # WER: ตัดคำไทยก่อน แล้วให้ jiwer เทียบแบบ whitespace-split ตามปกติ
    ref_tokenized = thai_tokenize_for_wer(ref_norm)
    hyp_tokenized = thai_tokenize_for_wer(hyp_norm)
    wer_output = jiwer.process_words(ref_tokenized, hyp_tokenized)

    # CER: เทียบระดับตัวอักษรตรง ๆ ไม่ต้องตัดคำ
    cer_output = jiwer.process_characters(ref_norm, hyp_norm)

    return wer_output, cer_output


def compute_rtf(audio_duration_sec: Optional[float], processing_time_sec: Optional[float]) -> Optional[float]:
    if not audio_duration_sec or processing_time_sec is None or audio_duration_sec <= 0:
        return None
    return round(processing_time_sec / audio_duration_sec, 4)


def evaluate(transcript_json_path: str, reference_txt_path: str) -> MetricResult:
    """
    transcript_json_path: path ไปยัง *_transcript.json ที่ module1_stt.py สร้างไว้
                           (ต้องมี key: full_transcript, audio_duration_sec, processing_time_sec)
    reference_txt_path:   path ไปยังไฟล์ .txt ที่พิมพ์ transcript ที่ถูกต้องไว้เอง
    """
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hypothesis = data.get("full_transcript", "")
    audio_duration = data.get("audio_duration_sec")
    processing_time = data.get("processing_time_sec")

    with open(reference_txt_path, "r", encoding="utf-8") as f:
        reference = f.read()

    if not reference.strip():
        raise ValueError(f"ไฟล์ reference ว่างเปล่า: {reference_txt_path}")
    if not hypothesis.strip():
        raise ValueError(f"full_transcript ใน {transcript_json_path} ว่างเปล่า")

    wer_output, cer_output = compute_wer_cer(reference, hypothesis)
    rtf = compute_rtf(audio_duration, processing_time)
    diff_report = build_diff_report(wer_output)

    return MetricResult(
        wer=round(wer_output.wer, 4),
        cer=round(cer_output.cer, 4),
        rtf=rtf,
        wer_substitutions=wer_output.substitutions,
        wer_deletions=wer_output.deletions,
        wer_insertions=wer_output.insertions,
        wer_hits=wer_output.hits,
        cer_substitutions=cer_output.substitutions,
        cer_deletions=cer_output.deletions,
        cer_insertions=cer_output.insertions,
        cer_hits=cer_output.hits,
        diff_report=diff_report,
    )


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="คำนวณ WER, CER, RTF จาก transcript JSON (module1_stt.py) เทียบกับ reference text"
    )
    parser.add_argument("transcript_json", help="path ไปยัง *_transcript.json")
    parser.add_argument("reference_txt", help="path ไปยังไฟล์ reference (.txt)")
    parser.add_argument("--out", default=None, help="path สำหรับบันทึกผลลัพธ์เป็น JSON (ไม่บังคับ)")
    args = parser.parse_args()

    result = evaluate(args.transcript_json, args.reference_txt)

    print(f"CER: {result.cer * 100:.2f}%  (hits={result.cer_hits}, sub={result.cer_substitutions}, "
          f"del={result.cer_deletions}, ins={result.cer_insertions})  ★ metric หลัก ★")
    print(f"WER: {result.wer * 100:.2f}%  (hits={result.wer_hits}, sub={result.wer_substitutions}, "
          f"del={result.wer_deletions}, ins={result.wer_insertions})  (ข้อมูลประกอบ อาจสูงเกินจริงถ้ามีคำทับศัพท์)")
    if result.rtf is not None:
        print(f"RTF: {result.rtf}  ({'เร็วกว่า' if result.rtf < 1 else 'ช้ากว่า'} real-time)")
    else:
        print("RTF: ไม่สามารถคำนวณได้ (ไม่พบ audio_duration_sec/processing_time_sec ใน JSON)")

    print("\n=== รายละเอียดจุดที่ผิด (diff) ===")
    print(result.diff_report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"\nบันทึกผลลัพธ์ที่: {args.out}")


if __name__ == "__main__":
    main()
