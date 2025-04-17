import cv2
import torch
import numpy as np

# Load the image
image = cv2.imread('monodepth2/assets/test_image2.jpg')
print("Image shape:", image.shape)

# Load the disparity map
disparity_map = np.load('monodepth2/assets/test_image2_disp.npy')  # (1,1,H,W)
disparity_map = np.squeeze(disparity_map)  # (H,W)
disparity_map = cv2.resize(disparity_map, (image.shape[1], image.shape[0]))  # Resize to match image
print("Disparity shape:", disparity_map.shape)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
results = model(image)

# Convert results to tensor
detections = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)
print("Detections:", detections)
print("Number of detections:", len(detections))

# Iterate over detections
for i, detection in enumerate(detections):
    x1, y1, x2, y2, conf, cls = detection
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Ensure crop box is within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(disparity_map.shape[1] - 1, x2), min(disparity_map.shape[0] - 1, y2)

    # Crop disparity map for the detected object
    disp_crop = disparity_map[y1:y2, x1:x2]
    
    if disp_crop.size == 0:
        print(f"Detection {i} skipped: empty crop.")
        continue

    # Calculate average depth from disparity (inverse relationship)
    avg_disparity = np.mean(disp_crop)
    depth = 1 / (avg_disparity + 1e-6)  # Add small epsilon to avoid division by zero

    # Draw bounding box and label on the image
    label = f"{model.names[int(cls)]} {depth:.2f}m"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show final result
display_image = cv2.resize(image, (1280, 720))  # or any size you prefer
cv2.imshow("Detections with Depth", image)
cv2.imwrite("detections_with_depth.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
