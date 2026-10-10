import os
import tensorflow as tf

# 限制 TensorFlow 显存并使用异步分配器
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from deepface import DeepFace

# ⚠️ 请在这里准备两张照片
# IMG1 可以继续用你刚才那张，IMG2 找一张你其他的照片，或者朋友的照片
IMG1 = "face_image/ch.jpg"
IMG2 = "face_image/lj.jpg"  # 记得把第二张图放到项目里并修改这里的名字！

if not os.path.exists(IMG1) or not os.path.exists(IMG2):
    print("❌ 错误：请确保两张照片路径都正确！")
else:
    print("🚀 正在提取两张照片的数学向量（Embedding）并进行比对...")
    try:
        # verify 会自动提取两张图的人脸特征并计算它们的欧氏距离
        result = DeepFace.verify(
            img1_path=IMG1,
            img2_path=IMG2,
            model_name="Facenet512",  # 默认经典的 Embedding 模型
            detector_backend="opencv"
        )

        # 强行释放显存
        tf.keras.backend.clear_session()

        print("\n" + "=" * 30 + " 比对结果 " + "=" * 30)
        print(f"这两张照片是同一个人吗？ -> 【{'是' if result['verified'] else '不是'}】")
        print(f"人脸特征空间距离 (Distance): {result['distance']:.4f}")
        print(f"判定同一人的最高临界值 (Threshold): {result['threshold']}")
        print("💡 提示：距离(Distance)越接近 0，说明长得越像！")
        print("=" * 68)

    except Exception as e:
        print(f"❌ 运行出错: {e}")