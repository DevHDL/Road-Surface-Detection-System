import cv2
import numpy as np
import datetime
from rknnlite.api import RKNNLite

IMG_SIZE = 640
CAM_WIDTH = 640
CAM_HEIGHT = 640
CLASSES = ("dry", "ice", "snow", "wet")

# decice tree for rk356x/rk3588
RK3588_RKNN_MODEL = 'res/yolov5_non_quant.rknn'

def draw_detections(frame, outputs, class_names, conf_threshold):
    """
    프레임에 YOLOv5 detection 결과를 표시하는 함수.

    Args:
        frame: OpenCV 프레임 (numpy array).
        outputs: YOLOv5 모델 outputs (list type, shape: (1, 1, 22500, 9)).
        class_names: 클래스 이름 리스트.
        conf_threshold: detection confidence 임계값.

    Returns:
        processed_frame: detection이 표시된 OpenCV 프레임.
    """
    processed_frame = frame.copy()
    output = outputs[0] # list type 이므로 첫 번째 element
    output_reshaped = output.reshape(-1, 9) # (22500, 9) 로 reshape

    H, W, _ = frame.shape

    for detection in output_reshaped:
        confidence = detection[4] # objectness score (5번째 값)
        #print(detection[:4])

        if confidence > conf_threshold:
            # bounding box 좌표 (center x, center y, width, height)
            center_x, center_y, w, h = detection[0], detection[1], detection[2], detection[3]

            x_min = int(center_x - w / 2)
            y_min = int(center_y - h / 2)
            x_max = int(center_x + w / 2)
            y_max = int(center_y + h / 2)
            
            # objectness score 와 class probabilities
            scores = detection[5:]
            class_id = np.argmax(scores) # 가장 높은 확률을 가진 클래스 ID
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id) # 클래스 이름 or class id
            label = f"{class_name}: {confidence:.2f}"
            print(w)

            # bounding box 및 label draw
            color = (0, 255, 0)
            cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(processed_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return processed_frame

if __name__ == '__main__':
    conf_threshold = 0.6
    rknn_model = RK3588_RKNN_MODEL
    rknn_lite = RKNNLite(verbose=False, verbose_file='./inference.log')

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # init video stream
    vs = cv2.VideoCapture('assets/original_video.mp4')
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

    if not vs.isOpened():
        print('Cannot capture from camera. Exiting.')
        quit()

    while True:
        start = datetime.datetime.now()
        ret, frame = vs.read() 
        if not ret: break
        frame_shape = frame.shape[:2]

        # Inference
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        exp_frame = np.expand_dims(img_rgb, axis=0)
        outputs = rknn_lite.inference(inputs=[frame], data_format='nhwc')
        
        processed_frame = draw_detections(frame, outputs, CLASSES, conf_threshold)

        end = datetime.datetime.now()
        total = (end-start).total_seconds()
        fps = f'FPS : {1 / total:.2f}'

        cv2.putText(processed_frame, fps, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow('frame', processed_frame)
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    rknn_lite.release()
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()