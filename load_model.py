import sys
sys.path.insert(0, r'C:\tf')  # Add custom TensorFlow installation path

import cv2
import numpy as np
from tensorflow.keras import models

MODEL_PATH = "tf-cnn-model.h5"

def predict_digit(image_path):
    
    # load model (compile=False to avoid compatibility issues with newer Keras)
    model = models.load_model(MODEL_PATH, compile=False)
    print("[INFO] Loaded model from disk.")

    image = cv2.imread(image_path, 0)      
    image1 = cv2.resize(image, (28,28))    # Resize to 28x28
    image2 = image1.reshape(1,28,28,1)

    # cv2.imshow('digit', image1 )  # Commented out for non-GUI environment
    print(f"[INFO] Processing image: {image_path}")
    print(f"[INFO] Image shape after preprocessing: {image2.shape}")
    pred = np.argmax(model.predict(image2), axis=-1)
    return pred[0]    

def main(image_path):
    predicted_digit = predict_digit(image_path)
    print('Predicted Digit: {}'.format(predicted_digit))
 
if __name__ == "__main__":
    try:
        main(image_path = sys.argv[1])
    except Exception as e:
        print(f'[ERROR]: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
