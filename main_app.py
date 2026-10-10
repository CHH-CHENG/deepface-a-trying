import cv2
import time
import tensorflow as tf
from camera_stream import CameraStream
from face_engine import FaceEngine


def main():
    # 初始化多线程摄像头与 AI 引擎
    cap = CameraStream(src=0).start()
    engine = FaceEngine(db_path="face_image")
    print("按 【ESC】 或 【Q】 键退出。")

    fps_start_time = time.time()
    fps_counter = 0
    fps_text = "FPS: 0"

    # 存储上一帧的检测结果，防止画面闪烁（做一个简单的平滑帧缓冲）
    last_results = []
    frame_count = 0

    while True:
        grabbed, frame = cap.read()
        if not grabbed or frame is None:
            continue

        frame_count += 1

        # 🎬 性能优化策略：每隔 3 帧让 GPU 预测一次，其余时间沿用上一帧红框。
        # 这样既能降低显存负荷，又能极大地提升显示画面的顺畅度（FPS飙升）
        if frame_count % 3 == 0:
            last_results = engine.process_frame(frame)
            # 及时释放显存碎片
            tf.keras.backend.clear_session()

        # 🎨 开始在画面上画框和写字
        for res in last_results:
            x, y, w, h = res["box"]
            identity = res["identity"]
            emotion = res["emotion"]

            # 根据是否认识这个人，切换红框（不认识）或绿框（认识）
            box_color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)

            # 画人脸矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # 渲染半透明标签背景
            label = f"{identity} | {emotion}"
            cv2.rectangle(frame, (x, y - 25), (x + w, y), box_color, cv2.FILLED)
            cv2.putText(frame, label, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 计算并绘制实时 FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps_text = f"FPS: {int(fps_counter / (time.time() - fps_start_time))}"
            fps_counter = 0
            fps_start_time = time.time()
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 弹出渲染大屏
        cv2.imshow("DeepFace Real-Time Monitor", frame)

        # 监听键盘事件
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 或 Q 键退出
            break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()