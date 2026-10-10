import os
import sys

# 【核心修复：限制 TensorFlow 狂吃显存】
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # 使用更高效的异步显存分配器

import cv2
import tensorflow as tf
from deepface import DeepFace

# 再次确保开启显存自适应增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMAGE_PATH = "face_image/chengh.jpg"  # 已经看到你项目里有这张图了

if not os.path.exists(IMAGE_PATH):
    print(f"❌ 错误：未找到照片【{IMAGE_PATH}】")
else:
    print("🚀 正在使用 GPU 优化模式分析照片...")
    try:
        # 为了防止 2G 显存一次性加载 3 个模型崩掉，我们采取“分而治之”的保守策略
        # 逐个分析，并及时清理后端缓存

        print("\n[1/3] 正在分析面部表情...")
        res_emotion = DeepFace.analyze(img_path=IMAGE_PATH, actions=['emotion'], detector_backend='opencv')
        tf.keras.backend.clear_session()  # 强行清空显存缓存

        print("[2/3] 正在预测性别...")
        res_gender = DeepFace.analyze(img_path=IMAGE_PATH, actions=['gender'], detector_backend='opencv')
        tf.keras.backend.clear_session()

        print("[3/3] 正在预测年龄...")
        res_age = DeepFace.analyze(img_path=IMAGE_PATH, actions=['age'], detector_backend='opencv')
        tf.keras.backend.clear_session()

        # 聚合结果
        print("\n" + "=" * 30 + " 最终识别结果 " + "=" * 30)

        print(f"主要表情 (Dominant Emotion): {res_emotion[0]['dominant_emotion']}")
        print(f"性别 (Gender): {res_gender[0]['dominant_gender']}")
        print(f"年龄 (Age): {res_age[0]['age']} 岁")
        print("=" * 68)

    except Exception as e:
        print(f"❌ 运行出错，错误信息: {e}")