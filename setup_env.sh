#!/bin/bash
# วิธีใช้: bash setup_env.sh

set -e

echo "==> สร้าง virtual environment ชื่อ 'interview_ai_env'"
python3 -m venv interview_ai_env

echo "==> Activate environment"
source interview_ai_env/bin/activate

echo "==> อัปเดต pip"
pip install --upgrade pip

echo "==> ติดตั้ง dependencies"
pip install faster-whisper moviepy librosa praat-parselmouth \
    mediapipe opencv-python deepface jiwer pythainlp \
    scipy scikit-learn numpy

echo "==> เสร็จแล้ว! ครั้งหน้าเปิด terminal ใหม่ ให้รัน:"
echo "    source interview_ai_env/bin/activate"
