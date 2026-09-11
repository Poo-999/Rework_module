"""
evaluate_eye_tracking.py
========================
ประเมินผลโมดูล Eye Tracking โดยเปรียบเทียบกับ Ground Truth (CSV)
- โหลด Ground Truth จากไฟล์ CSV ที่สร้างด้วย annotation_tool.py
- รันโมดูล Eye Tracking เพื่อให้ได้ Predictions
- จับคู่ตาม frame index
- คำนวณ Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)

# Import โมดูล Eye Tracking ของคุณ (แก้ไขพาธให้ถูกต้อง)
# สมมติว่าไฟล์ eye_track_merge.py อยู่ในโฟลเดอร์เดียวกัน
sys.path.append(os.path.dirname(__file__))
from eye_track_merge import EyeTrackingVisualizer, analyze_frame_attention, GazeStabilizer


# ============================================================
# 1. โหลด Ground Truth จาก CSV
# ============================================================
def load_ground_truth(csv_path: str) -> dict:
    """
    โหลด Ground Truth ที่สร้างจาก annotation_tool.py
    คืนค่า: dict {frame_idx: label} โดย label = 1 (มอง) หรือ 0 (ไม่มอง)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # ตรวจสอบคอลัมน์
    if 'frame_idx' not in df.columns:
        # ถ้าไม่มี frame_idx ให้ใช้ timestamp คำนวณ
        print("⚠️ CSV ไม่มีคอลัมน์ frame_idx กำลังคำนวณจาก timestamp...")
        # สมมติว่า FPS = 30 (หรือดึงจากวิดีโอจริง)
        fps = 30.0
        if 'fps' in df.columns:
            fps = df['fps'].iloc[0]
        df['frame_idx'] = (df['timestamp_sec'] * fps).astype(int)
    
    # สร้าง dict
    gt_dict = dict(zip(df['frame_idx'], df['label']))
    print(f"✅ โหลด Ground Truth: {len(gt_dict)} เฟรม")
    print(f"   - มองกล้อง (1): {sum(1 for v in gt_dict.values() if v == 1)} เฟรม")
    print(f"   - ไม่มอง (0): {sum(1 for v in gt_dict.values() if v == 0)} เฟรม")
    return gt_dict


# ============================================================
# 2. รันโมดูล Eye Tracking เพื่อให้ได้ Predictions
# ============================================================
def get_predictions(video_path: str, target_frames: list = None) -> dict:
    """
    รันโมดูล Eye Tracking บนวิดีโอ แล้วคืนค่า predictions สำหรับเฟรมที่ต้องการ
    
    Args:
        video_path: พาธไฟล์วิดีโอ
        target_frames: list ของ frame index ที่ต้องการ prediction 
                       (ถ้าเป็น None จะทำทุกเฟรม)
    
    Returns:
        dict: {frame_idx: predicted_label} (0 หรือ 1)
    """
    print(f"🔍 กำลังประมวลผลวิดีโอ: {video_path}")
    
    # สร้าง instance ของโมดูล Eye Tracking
    # ใช้ face_landmarker.task ที่คุณมี
    visualizer = EyeTrackingVisualizer(model_path="face_landmarker.task")
    stabilizer = GazeStabilizer(smooth_window=5, debounce_frames=3)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"ไม่สามารถเปิดวิดีโอ: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ถ้าไม่ระบุ target_frames ให้ทำทุกเฟรม
    if target_frames is None:
        target_frames = set(range(total_frames))
    else:
        target_frames = set(target_frames)
    
    predictions = {}
    frame_idx = 0
    
    print(f"   จำนวนเฟรมทั้งหมด: {total_frames}")
    print(f"   จำนวนเฟรมที่ต้องการ predict: {len(target_frames)}")
    
    # ใช้ Mediapipe Face Landmarker (ต้อง import)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # ถ้าเฟรมนี้อยู่ใน target_frames หรือทำทุกเฟรม
        if frame_idx in target_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            pred_label = 0  # default = ไม่มอง
            if result.face_blendshapes and result.facial_transformation_matrixes:
                blendshapes = result.face_blendshapes[0]
                matrix = np.array(result.facial_transformation_matrixes[0])
                landmarks = result.face_landmarks[0]
                
                state = analyze_frame_attention(blendshapes, matrix, landmarks, width, height)
                
                # ใช้ Stabilizer
                stabilized = stabilizer.update(
                    yaw=state["yaw"],
                    pitch=state["pitch"],
                    max_gaze_deviation=state["max_gaze_deviation"],
                    is_blinking=state["is_blinking"],
                )
                is_looking = stabilized["is_looking_camera"]
                pred_label = 1 if is_looking else 0
            
            predictions[frame_idx] = pred_label
        
        frame_idx += 1
        
        # แสดง progress ทุก 100 เฟรม
        if frame_idx % 100 == 0:
            print(f"   ประมวลผลเฟรมที่ {frame_idx}/{total_frames}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ ได้ Predictions: {len(predictions)} เฟรม")
    looking_count = sum(1 for v in predictions.values() if v == 1)
    print(f"   - โมเดลทำนายว่ามอง: {looking_count} เฟรม")
    print(f"   - โมเดลทำนายว่าไม่มอง: {len(predictions) - looking_count} เฟรม")
    
    return predictions


# ============================================================
# 3. จับคู่และประเมินผล
# ============================================================
def evaluate(gt_dict: dict, pred_dict: dict) -> dict:
    """
    จับคู่ Ground Truth กับ Predictions ตาม frame index
    แล้วคำนวณเมตริกทั้งหมด
    """
    # หา frame ที่มีทั้ง GT และ Prediction
    common_frames = set(gt_dict.keys()) & set(pred_dict.keys())
    
    if not common_frames:
        raise ValueError("❌ ไม่มีเฟรมที่ตรงกันระหว่าง Ground Truth กับ Predictions!")
    
    print(f"\n📊 พบเฟรมที่ตรงกัน: {len(common_frames)} เฟรม")
    
    y_true = [gt_dict[f] for f in common_frames]
    y_pred = [pred_dict[f] for f in common_frames]
    
    # คำนวณเมตริก
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Classification Report (ละเอียด)
    report = classification_report(y_true, y_pred, target_names=['Not Looking', 'Looking'], output_dict=True)
    
    result = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
        "sample_size": len(common_frames),
        "classification_report": report,
    }
    
    return result


# ============================================================
# 4. ฟังก์ชันหลัก (Main)
# ============================================================
def main(video_path: str, gt_csv_path: str):
    """
    Pipeline ประเมินผลแบบครบวงจร
    """
    print("=" * 60)
    print("🧪 เริ่มกระบวนการประเมินผล Eye Tracking Module")
    print("=" * 60)
    
    # ขั้นตอนที่ 1: โหลด Ground Truth
    print("\n[1] โหลด Ground Truth")
    gt_dict = load_ground_truth(gt_csv_path)
    
    # ขั้นตอนที่ 2: รันโมดูลเพื่อให้ได้ Predictions (เฉพาะเฟรมที่มี GT)
    print("\n[2] รับ Predictions จากโมดูล Eye Tracking")
    target_frames = list(gt_dict.keys())
    pred_dict = get_predictions(video_path, target_frames)
    
    # ขั้นตอนที่ 3: ประเมินผล
    print("\n[3] ประเมินผล")
    result = evaluate(gt_dict, pred_dict)
    
    # แสดงผล
    print("\n" + "=" * 60)
    print("📊 ผลการประเมิน Eye Tracking Module")
    print("=" * 60)
    print(f"✅ Accuracy : {result['accuracy']:.4f}")
    print(f"✅ Precision: {result['precision']:.4f}")
    print(f"✅ Recall   : {result['recall']:.4f}")
    print(f"✅ F1 Score : {result['f1_score']:.4f}")
    print(f"\n📋 Confusion Matrix:")
    cm = result['confusion_matrix']
    print(f"   TP (มองถูก) : {cm['true_positive']}")
    print(f"   FN (มองแต่โมเดลว่าอาจไม่) : {cm['false_negative']}")
    print(f"   FP (ไม่มองแต่โมเดลว่ามอง) : {cm['false_positive']}")
    print(f"   TN (ไม่มองถูก) : {cm['true_negative']}")
    print(f"\n📊 จำนวนตัวอย่างทั้งหมด: {result['sample_size']}")
    
    # บันทึกผลลัพธ์เป็น JSON
    import json
    output_json = Path(gt_csv_path).stem + "_evaluation_result.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        # แปลง numpy int64 เป็น int ปกติ
        json_result = {
            "accuracy": result['accuracy'],
            "precision": result['precision'],
            "recall": result['recall'],
            "f1_score": result['f1_score'],
            "confusion_matrix": result['confusion_matrix'],
            "sample_size": result['sample_size'],
        }
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 บันทึกผลลัพธ์ที่: {output_json}")
    
    return result


# ============================================================
# 5. ตัวอย่างการใช้งาน
# ============================================================
if __name__ == "__main__":
    # รับ Path จากผู้ใช้ (ปลอดภัย)
    video_path = r"D:\Modeling\Data\blinking.mp4"
    
    gt_csv_path = r"D:\Modeling\Data\blinking_ground_truth_eye.csv"
    
    if not os.path.exists(video_path):
        print(f"❌ ไม่พบไฟล์วิดีโอ: {video_path}")
        exit(1)
    
    if not os.path.exists(gt_csv_path):
        print(f"❌ ไม่พบไฟล์ CSV: {gt_csv_path}")
        print("   ให้ค้นหาไฟล์ CSV ที่สร้างจาก annotation_tool.py")
        # แสดงไฟล์ในโฟลเดอร์ปัจจุบัน
        print("\n📁 ไฟล์ CSV ที่พบในโฟลเดอร์นี้:")
        for f in os.listdir():
            if f.endswith(".csv"):
                print(f"   - {f}")
        exit(1)
    
    try:
        result = main(video_path, gt_csv_path)
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()