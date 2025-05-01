import torch
import cv2
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    conf_threshold = 0.6
    fps_array = []
    model_path = 'model/best.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)
    model.to(device)
    
    vs = cv2.VideoCapture('assets/original_video.mp4')

    if not vs.isOpened():
        print('Cannot capture from camera. Exiting.')
        quit()

    while True:
        start = datetime.datetime.now()
        ret, frame = vs.read() 
        frame = cv2.resize(frame, dsize=(640,640))
        if not ret: break

        # Inference
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = model(img_rgb)

        for *xyxy, conf, cls in outputs.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        end = datetime.datetime.now()
        total = (end-start).total_seconds()
        fps = f'FPS : {1 / total:.2f}'
        fps_array.append(round(1/total, 3))

        cv2.putText(frame, fps, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow('frame', frame)
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # do a bit of cleanup
    print(f'fps average : {np.mean(fps_array):.2f}')
    cv2.destroyAllWindows()
    vs.release()