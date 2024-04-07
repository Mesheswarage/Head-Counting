import cv2
import numpy as np
import threading
from tensorflow import keras

# Global variables
frame = None
mask = None
exit_flag = False

# Load the model
model = keras.models.load_model('F:/medhavi/EDU/Final_Project/Model/crowd_detection_model_V2_acc_(0.61).h5', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy')

def preprocess_image(image):
    resized_image = cv2.resize(image, (256, 256))
    processed_image = resized_image / 255.0  
    processed_image = np.expand_dims(processed_image, axis=0)  
    return processed_image

def predict_mask(frame):
    global mask
    processed_frame = preprocess_image(frame)
    mask = model.predict(processed_frame)[0]  

def inference_worker(video_capture):
    global frame, exit_flag
    while not exit_flag:
        ret, frame = video_capture.read()
        if not ret:
            break
        predict_mask(frame)

def main():
    global frame, mask, exit_flag
    video_capture = cv2.VideoCapture('E:/JHU/data/20240219_162022.mp4')
    if not video_capture.isOpened():
        print("Error: Could not open video file")
        return

    inference_thread = threading.Thread(target=inference_worker, args=(video_capture,))
    inference_thread.start()

    while True:
        if frame is not None:
            resized_frame = cv2.resize(frame, (256, 256))  # Resize frame to 256x256

            if mask is not None:
                mask_uint8 = (mask * 255).astype(np.uint8)
                ret, thresh = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                cv2.putText(resized_frame, f'Contours: {len(contours)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Original Video', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break

    inference_thread.join()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
