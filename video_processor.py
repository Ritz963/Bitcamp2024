import cv2
from facenet_pytorch import MTCNN
import torch
import time

def open_webcam():
    # Set device for computations (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize the MTCNN face detector
    mtcnn = MTCNN(keep_all=True, device=device)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(1) 

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Set properties for webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    start_time = None
    detected_for = 3  # seconds
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame_rgb)

        # Draw bounding box around detected faces
        if boxes is not None and len(boxes) > 0:
            # Calculate the area and find the largest face
            max_area = 0
            max_box = None
            for box in boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    max_box = box

            if max_box is not None:
                # Expand the box by 'margin' pixels
                margin = 20
                x1 = max(int(max_box[0]) - margin, 0)
                y1 = max(int(max_box[1]) - margin, 0)
                x2 = min(int(max_box[2]) + margin, frame.shape[1])
                y2 = min(int(max_box[3]) + margin, frame.shape[0])
                cv2.rectangle(frame, (int(max_box[0]) - margin - 10, int(max_box[1]) - margin - 10), (int(max_box[2]) + margin + 10, int(max_box[3]) + margin + 10), (255, 0, 0), 2)

                if start_time is None:
                    start_time = time.time()  # Start the timer when a face is first detected

                elapsed_time = time.time() - start_time
                if elapsed_time >= detected_for and not saved:
                    closest_face = frame[y1:y2, x1:x2]
                    cv2.imwrite('closest_face.jpg', closest_face)
                    print("Saved the closest face to 'closest_face.jpg'")
                    saved = True  # Set flag to prevent multiple saves
                    break
        else:
            start_time = None  # Reset timer if no face is detected
            saved = False  # Reset save flag

        # Display the resulting frame
        cv2.imshow('Webcam - Face Detection', frame)

        # Wait for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    open_webcam()
