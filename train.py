from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
model = YOLO('ultralytics/cfg/models/new/')  # 从YAML构建并传输权重

if __name__ == '__main__':
    # 训练模型
    results = model.train(data='', epochs=1, imgsz=640, batch=4, workers=1)
    # 模型验证
    #model.val()

