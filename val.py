import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('weights/best.pt')
    model.val(data='', split='test', save_json=True)