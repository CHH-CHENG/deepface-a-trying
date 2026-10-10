import os
import numpy as np

# 强制注入你的 GPU 黄金律令
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from deepface import DeepFace


class FaceEngine:
    def __init__(self, db_path="face_image"):
        self.db_path = db_path
        self.model_name = "Facenet512"  # 绝杀冒牌货的强悍模型
        self.detector_backend = "opencv"  # 换回通用且实时速度极快的 opencv 后端
        self.threshold = 0.3  # FaceNet512 的严格及格线
        print(f"加载 [{self.model_name}] 与 Emotion 模型...")

    def process_frame(self, frame):
        """核心处理逻辑：输入一帧，返回检测到的人脸结构化数据"""
        results_out = []
        try:
            # 1. 目标检测 + 表情分析
            # Enforce_detection=False 防止画面中没人时直接崩掉
            objs = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            # 2. 身份识别 (与你的 face_image 文件夹比对)
            dfs = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            # 3. 结构化拼装数据
            for i, obj in enumerate(objs):
                region = obj["region"]
                if region["w"] < 40 or region["h"] < 40:  # 过滤背景杂音
                    continue

                box = [region["x"], region["y"], region["w"], region["h"]]
                emotion = obj["dominant_emotion"]

                # 解析身份
                identity = "Unknown"
                if len(dfs) > i and not dfs[i].empty:
                    # 拿到亲和度最高的第一条匹配记录
                    match = dfs[i].iloc[0]
                    if match["distance"] < self.threshold:
                        # 从绝对路径提取文件名作为人名 (例如: face_image/ch.jpg -> ch)
                        file_name = os.path.basename(match["identity"])
                        identity = os.path.splitext(file_name)[0]

                results_out.append({
                    "box": box,
                    "emotion": emotion,
                    "identity": identity
                })
        except Exception as e:
            # 打印明显的错误，方便我们定位
            print(f"引擎内部处理帧出错: {e}")
            pass

        return results_out