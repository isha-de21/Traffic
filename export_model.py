from ultralytics import YOLO

def main():
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    print("Exporting to ONNX format...")
    try:
        model.export(format='onnx')
        print("ONNX export successful.")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        
    print("Exporting to NCNN format...")
    try:
        model.export(format='ncnn')
        print("NCNN export successful.")
    except Exception as e:
        print(f"Error exporting to NCNN: {e}")

if __name__ == "__main__":
    main()
