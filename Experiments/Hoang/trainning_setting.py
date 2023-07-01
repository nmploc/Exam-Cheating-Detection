from argparse import Namespace
import train_yolov8
# Tạo đối tượng args từ training_settings
args = Namespace(**train_yolov8.training_settings)

# Sử dụng các giá trị trong args
print("Number of epochs:", args.epochs)
print("Batch size:", args.batch)
print("Learning rate:", args.lr0)