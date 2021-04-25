import cv2 
import numpy as np 
import onnxruntime as ort 

onnx_model = 'ASL.onnx'

def crop(frame):
    h, w, _ = frame.shape 
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]

def main():
    labs = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    m = 0.485 * 255
    s = 0.229 * 255

    ort_session = ort.InferenceSession(onnx_model)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame = crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x =  (cv2.resize(frame, (28, 28)) - m) / s
        x = x.reshape(1,1,28,28).astype(np.float32)
        y = ort_session.run(None, {'input':x})[0]
        
        index = np.argmax(y, axis = 1)
        label = labs[int(index)]

        cv2.putText(frame, label, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness = 2)
        cv2.imshow('Live Sign Language Translator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()