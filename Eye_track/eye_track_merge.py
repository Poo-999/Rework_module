import os
import cv2
import time
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# 1. ฟังก์ชันคำนวณและวาดกราฟิก Visual Debugging (คงเดิม)
# ==============================================================================

def get_head_pose_angles(matrix_4x4: np.ndarray):
    """แปลง 4x4 Transformation Matrix เป็นมุม Euler (Pitch, Yaw, Roll)"""
    rot_matrix = matrix_4x4[:3, :3]
    r = Rotation.from_matrix(rot_matrix)
    pitch, yaw, roll = r.as_euler('xyz', degrees=True)
    return pitch, yaw, roll, rot_matrix


def draw_head_pose_axes(frame: np.ndarray, anchor_pt: tuple, rot_matrix: np.ndarray, length: int = 70):
    """วาดแกน 3 มิติ (RGB) ออกมาจากจุด Anchor (ปลายจมูก)"""
    cx, cy = anchor_pt
    x_axis = rot_matrix @ np.array([length, 0, 0])
    y_axis = rot_matrix @ np.array([0, length, 0])
    z_axis = rot_matrix @ np.array([0, 0, length])

    pt_x = (int(cx + x_axis[0]), int(cy - x_axis[1]))
    pt_y = (int(cx + y_axis[0]), int(cy - y_axis[1]))
    pt_z = (int(cx + z_axis[0]), int(cy - z_axis[1]))

    cv2.arrowedLine(frame, (cx, cy), pt_x, (0, 0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(frame, (cx, cy), pt_y, (0, 255, 0), 2, tipLength=0.2)
    cv2.arrowedLine(frame, (cx, cy), pt_z, (255, 0, 0), 3, tipLength=0.2)
    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)


def draw_hud_panel(frame: np.ndarray, state: dict, summary_stats: dict, perf_stats: dict):
    """
    วาด HUD เพิ่มเติมแสดง Performance (FPS, RTF) และสถิติ Head Movement
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (420, 400), (20, 20, 20), -1)  # ขยายอีกนิด
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Status Badge
    if state["is_blinking"]:
        status_text = "STATUS: BLINKING"
        badge_color = (0, 165, 255)
    elif state["is_looking_camera"]:
        status_text = "STATUS: LOOKING AT CAMERA"
        badge_color = (0, 255, 0)
    else:
        status_text = "STATUS: GAZE DIVERTED"
        badge_color = (0, 0, 255)

    cv2.rectangle(frame, (30, 30), (380, 70), badge_color, -1)
    cv2.putText(frame, status_text, (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (240, 240, 240)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)

    # สถานะรายเฟรม
    cv2.putText(frame, f"Head Yaw   : {state['yaw']:+.1f} deg", (35, 100), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Head Pitch : {state['pitch']:+.1f} deg", (35, 125), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Head Roll  : {state['roll']:+.1f} deg", (35, 150), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gaze Offset: {state['max_gaze_deviation']:.2f}", (35, 175), font, 0.52, white, 1, cv2.LINE_AA)

    cv2.line(frame, (30, 190), (380, 190), (80, 80, 80), 1)

    # สถิติภาพรวมเชิงพฤติกรรม (Behavioral)
    cv2.putText(frame, "--- BEHAVIORAL ---", (35, 215), font, 0.52, yellow, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Eye Contact: {summary_stats['eye_contact_ratio']*100:.1f} %", (35, 245), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Blink Rate : {summary_stats['blink_rate_bpm']:.1f} bpm", (35, 275), font, 0.52, white, 1, cv2.LINE_AA)
    
    cv2.line(frame, (30, 295), (380, 295), (80, 80, 80), 1)

    # สถิติประสิทธิภาพ (Performance) + การเคลื่อนไหวศีรษะ
    cv2.putText(frame, "--- PERFORMANCE ---", (35, 320), font, 0.52, cyan, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Proc. FPS  : {perf_stats['processing_fps']:.1f} fps", (35, 350), font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"RTF        : {perf_stats['rtf']:.3f} x", (35, 375), font, 0.52, white, 1, cv2.LINE_AA)


def calculate_iris_gaze(landmarks, image_width: int, image_height: int):
    """คงเดิม"""
    def get_pt(idx):
        return np.array([landmarks[idx].x * image_width, landmarks[idx].y * image_height])

    r_outer = get_pt(33)
    r_inner = get_pt(133)
    r_iris = get_pt(468)
    r_width = np.linalg.norm(r_outer - r_inner)
    r_ratio_x = np.linalg.norm(r_iris - r_inner) / r_width if r_width > 0 else 0.5
    r_dev_x = abs(r_ratio_x - 0.50)

    l_inner = get_pt(362)
    l_outer = get_pt(263)
    l_iris = get_pt(473)
    l_width = np.linalg.norm(l_outer - l_inner)
    l_ratio_x = np.linalg.norm(l_iris - l_inner) / l_width if l_width > 0 else 0.5
    l_dev_x = abs(l_ratio_x - 0.50)

    r_top = get_pt(159)
    r_bottom = get_pt(145)
    r_height = np.linalg.norm(r_bottom - r_top)
    r_ratio_y = np.linalg.norm(r_iris - r_top) / r_height if r_height > 0 else 0.5
    r_dev_y = abs(r_ratio_y - 0.50)

    avg_dev_x = (r_dev_x + l_dev_x) / 2.0
    avg_dev_y = r_dev_y
    max_gaze_deviation = max(avg_dev_x, avg_dev_y * 0.8)

    return {
        "max_gaze_deviation": round(float(max_gaze_deviation), 3),
        "r_iris_center": (int(r_iris[0]), int(r_iris[1])),
        "l_iris_center": (int(l_iris[0]), int(l_iris[1])),
    }


def analyze_frame_attention(blendshape_list, head_matrix, landmarks, width: int, height: int):
    """คงเดิม"""
    scores = {b.category_name: b.score for b in blendshape_list}
    avg_blink = (scores.get('eyeBlinkLeft', 0.0) + scores.get('eyeBlinkRight', 0.0)) / 2.0
    is_blinking = avg_blink > 0.5

    pitch, yaw, roll, rot_matrix = get_head_pose_angles(head_matrix)
    is_head_centered = (abs(yaw) <= 20.0) and (abs(pitch) <= 22.0)

    iris_info = calculate_iris_gaze(landmarks, width, height)
    max_gaze_deviation = iris_info["max_gaze_deviation"]
    is_eye_centered = max_gaze_deviation < 0.14

    is_looking_camera = is_head_centered and is_eye_centered and (not is_blinking)

    return {
        "is_blinking": is_blinking,
        "is_looking_camera": is_looking_camera,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "rot_matrix": rot_matrix,
        "max_gaze_deviation": max_gaze_deviation,
        "avg_blink": avg_blink,
        "r_iris_center": iris_info["r_iris_center"],
        "l_iris_center": iris_info["l_iris_center"],
    }


class GazeStabilizer:
    """คงเดิม"""
    def __init__(self, smooth_window: int = 5, debounce_frames: int = 3):
        self.smooth_window = smooth_window
        self.debounce_frames = debounce_frames
        self._yaw_buf = deque(maxlen=smooth_window)
        self._pitch_buf = deque(maxlen=smooth_window)
        self._gaze_buf = deque(maxlen=smooth_window)
        self.stable_state = False
        self._pending_state = None
        self._pending_count = 0

    def update(self, yaw: float, pitch: float, max_gaze_deviation: float, is_blinking: bool) -> dict:
        if not is_blinking:
            self._yaw_buf.append(yaw)
            self._pitch_buf.append(pitch)
            self._gaze_buf.append(max_gaze_deviation)

        smoothed_yaw = sum(self._yaw_buf) / len(self._yaw_buf) if self._yaw_buf else yaw
        smoothed_pitch = sum(self._pitch_buf) / len(self._pitch_buf) if self._pitch_buf else pitch
        smoothed_gaze = sum(self._gaze_buf) / len(self._gaze_buf) if self._gaze_buf else max_gaze_deviation

        is_head_centered = (abs(smoothed_yaw) <= 20.0) and (abs(smoothed_pitch) <= 22.0)
        is_eye_centered = smoothed_gaze < 0.14
        raw_candidate = is_head_centered and is_eye_centered

        if raw_candidate == self.stable_state:
            self._pending_state = None
            self._pending_count = 0
        else:
            if self._pending_state == raw_candidate:
                self._pending_count += 1
            else:
                self._pending_state = raw_candidate
                self._pending_count = 1
            if self._pending_count >= self.debounce_frames:
                self.stable_state = raw_candidate
                self._pending_state = None
                self._pending_count = 0

        actual_looking = self.stable_state and (not is_blinking)

        return {
            "is_looking_camera": actual_looking,
            "smoothed_yaw": smoothed_yaw,
            "smoothed_pitch": smoothed_pitch,
            "smoothed_gaze_deviation": smoothed_gaze,
        }


# ==============================================================================
# 2. คลาสหลักที่อัปเกรดให้มี Performance Metrics และ Advanced Metrics
# ==============================================================================

class EyeTrackingVisualizer:
    def __init__(self, model_path: str = "face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.stabilizer = GazeStabilizer(smooth_window=5, debounce_frames=3)

    def process_and_render(self, video_path: str, output_path: str = None, show_window: bool = True):
        start_time = time.time()  # <--- เริ่มจับเวลา Performance

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"ไม่สามารถเปิดไฟล์วิดีโอได้: {video_path}")

        total_frames = 0
        valid_gaze_frames = 0
        looking_frames = 0
        blink_counter = 0
        blink_in_progress = False

        # --- ตัวแปรสำหรับ Advanced Metrics ใหม่ ---
        yaw_history = []
        pitch_history = []
        gaze_shift_count = 0
        prev_looking_state = None  # ใช้เช็คการเปลี่ยนสถานะ

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)

            if result.face_blendshapes and result.facial_transformation_matrixes:
                blendshapes = result.face_blendshapes[0]
                matrix = np.array(result.facial_transformation_matrixes[0])
                landmarks = result.face_landmarks[0]

                state = analyze_frame_attention(blendshapes, matrix, landmarks, width, height)

                # เก็บค่า Yaw/Pitch เพื่อคำนวณสถิติช่วงการขยับ
                yaw_history.append(state["yaw"])
                pitch_history.append(state["pitch"])

                # Temporal Smoothing
                stabilized = self.stabilizer.update(
                    yaw=state["yaw"],
                    pitch=state["pitch"],
                    max_gaze_deviation=state["max_gaze_deviation"],
                    is_blinking=state["is_blinking"],
                )
                state["is_looking_camera"] = stabilized["is_looking_camera"]

                # นับ Blink
                if state["is_blinking"]:
                    if not blink_in_progress:
                        blink_counter += 1
                        blink_in_progress = True
                else:
                    blink_in_progress = False
                    valid_gaze_frames += 1
                    if state["is_looking_camera"]:
                        looking_frames += 1

                # --- นับจำนวนครั้งที่เปลี่ยนสถานะการสบตา (Gaze Shifts) ---
                current_state = state["is_looking_camera"]
                if prev_looking_state is not None and current_state != prev_looking_state:
                    gaze_shift_count += 1
                prev_looking_state = current_state

                # คำนวณสถิติเพื่อแสดงผลสด
                current_duration = total_frames / fps if fps > 0 else 0
                current_ratio = looking_frames / valid_gaze_frames if valid_gaze_frames > 0 else 0.0
                current_bpm = (blink_counter / current_duration) * 60.0 if current_duration > 0 else 0.0

                live_summary = {
                    "duration_sec": current_duration,
                    "eye_contact_ratio": current_ratio,
                    "blink_count": blink_counter,
                    "blink_rate_bpm": current_bpm
                }

                # --- คำนวณ Performance แบบ Real-time เพื่อแสดงใน HUD ---
                elapsed = time.time() - start_time
                current_proc_fps = total_frames / elapsed if elapsed > 0 else 0
                current_rtf = elapsed / current_duration if current_duration > 0 else 0
                perf_stats = {
                    "processing_fps": current_proc_fps,
                    "rtf": current_rtf,
                }

                # วาดกราฟิก
                nose_pt = (int(landmarks[1].x * width), int(landmarks[1].y * height))
                draw_head_pose_axes(frame, nose_pt, state["rot_matrix"], length=80)
                draw_hud_panel(frame, state, live_summary, perf_stats)

            else:
                # กรณีหันหน้าหนีจนหลุดเฟรม (Face Not Found) 
                # ถือว่าไม่ได้สบตา และเป็นการเปลี่ยนสถานะด้วย (ถ้าก่อนหน้านี้สบตา)
                blink_in_progress = False
                valid_gaze_frames += 1
                current_state = False
                if prev_looking_state is not None and current_state != prev_looking_state:
                    gaze_shift_count += 1
                prev_looking_state = current_state

            if writer:
                writer.write(frame)

            if show_window:
                cv2.imshow("Eye & Head Pose Debug View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # --- สรุปผลลัพธ์ ---
        end_time = time.time()
        processing_time = end_time - start_time
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Behavioral Metrics (เดิม)
        eye_contact_ratio = looking_frames / valid_gaze_frames if valid_gaze_frames > 0 else 0.0
        blink_rate = (blink_counter / duration_sec) * 60.0 if duration_sec > 0 else 0.0

        # Advanced Behavioral Metrics (ใหม่)
        avg_abs_yaw = sum(abs(y) for y in yaw_history) / len(yaw_history) if yaw_history else 0.0
        max_abs_yaw = max(abs(y) for y in yaw_history) if yaw_history else 0.0
        avg_abs_pitch = sum(abs(p) for p in pitch_history) / len(pitch_history) if pitch_history else 0.0
        max_abs_pitch = max(abs(p) for p in pitch_history) if pitch_history else 0.0

        # Performance Metrics (ใหม่) -> เทียบเท่ากับ RTF ของ Whisper
        rtf = processing_time / duration_sec if duration_sec > 0 else 0.0
        avg_processing_fps = total_frames / processing_time if processing_time > 0 else 0.0

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print("ประมวลผลวิดีโอเสร็จสิ้น")

        # ส่งคืนผลลัพธ์แบบแบ่งหมวดหมู่
        return {
            "performance": {
                "video_duration_sec": round(duration_sec, 2),
                "processing_time_sec": round(processing_time, 2),
                "rtf": round(rtf, 4),                  # <-- Metric หลักเปรียบเทียบกับ STT
                "avg_processing_fps": round(avg_processing_fps, 1),
                "total_frames": total_frames,
            },
            "behavioral": {
                "eye_contact_ratio": round(eye_contact_ratio, 3),
                "blink_count": blink_counter,
                "blink_rate_bpm": round(blink_rate, 1),
                # Advanced Metrics สำหรับประเมินความมั่นใจ/สมาธิ
                "avg_abs_head_yaw": round(avg_abs_yaw, 2),      # ค่าเฉลี่ยการหันซ้าย-ขวา
                "max_abs_head_yaw": round(max_abs_yaw, 2),      # ค่าสูงสุดการหันซ้าย-ขวา
                "avg_abs_head_pitch": round(avg_abs_pitch, 2),  # ค่าเฉลี่ยการเงย-ก้ม
                "max_abs_head_pitch": round(max_abs_pitch, 2),  # ค่าสูงสุดการเงย-ก้ม
                "gaze_shift_count": gaze_shift_count,           # จำนวนครั้งที่ละสายตากลับมามอง
            }
        }


# ==============================================================================
# 3. ส่วนรันโค้ดและพิมพ์ผลลัพธ์
# ==============================================================================
if __name__ == "__main__":
    visualizer = EyeTrackingVisualizer(model_path="face_landmarker.task")
    
    result = visualizer.process_and_render(
        video_path=r"D:\Modeling\Data\stare.mp4",
        output_path="debug_output.mp4",
        show_window=True
    )
    
    print("\n" + "=" * 50)
    print("📊 ผลการวิเคราะห์โมดูล Eye Tracking & Head Pose (ฉบับสมบูรณ์):")
    print("=" * 50)
    
    print("\n[ PERFORMANCE METRICS ] (คล้าย RTF ของ Whisper)")
    for k, v in result["performance"].items():
        print(f"  {k}: {v}")
    
    print("\n[ BEHAVIORAL METRICS ] (ใช้วิเคราะห์บุคลิกภาพ)")
    for k, v in result["behavioral"].items():
        print(f"  {k}: {v}")
    print("=" * 50)