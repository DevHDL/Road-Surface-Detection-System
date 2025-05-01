import cv2
import numpy as np
import time
import threading
from rknnlite.api import RKNNLite

CAM_WIDTH = 640
CAM_HEIGHT = 640
RKNN_MODEL_PATH = 'res/road_model.rknn'
VIDEO_PATH = 0 # test video : 'assets/original_video.mp4'
CLASSES = ("dry", "ice", "snow", "wet")
CONF_THRESHOLD = 0.4

class VideoStream:
    def __init__(self, src="original.mp4"):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                return
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()

def draw_detections(frame, outputs, class_names, conf_threshold, iou_threshold=0.4):
    output = outputs[0]
    output_reshaped = output.reshape(-1, 9)

    boxes = []
    confidences = []
    class_ids = []

    # divide box, confidence, class_id
    for detection in output_reshaped:
        confidence = detection[4]
        # confidence threshold
        if confidence > conf_threshold:
            center_x, center_y, w, h = detection[0], detection[1], detection[2], detection[3]
            x_min = max(0, int(center_x - w / 2))
            y_min = max(0, int(center_y - h / 2))

            scores = detection[5:]
            class_id = np.argmax(scores)

            boxes.append([x_min, y_min, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # calculate NMS 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[i]
        x, y, w, h = box
        x_max = min(x+w, CAM_WIDTH)
        y_max = min(y+h, CAM_HEIGHT)
        class_id = class_ids[i]
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        label = f"{class_name}: {confidences[i]:.2f}"

        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

if __name__ == '__main__':
    conf_threshold = CONF_THRESHOLD
    rknn_model = RKNN_MODEL_PATH

    # load RKNN model
    rknn_lite = RKNNLite(verbose=False, verbose_file='./inference.log')

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
    cap = VideoStream(src=VIDEO_PATH).start()

    while not cap.stopped:
        ret, frame = cap.read() 
        if not ret: break

        # pre-processing
        frame = cv2.resize(frame, dsize=(CAM_WIDTH,CAM_HEIGHT))
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        exp_frame = np.expand_dims(img_rgb, axis=0)

        # inference
        outputs = rknn_lite.inference(inputs=[exp_frame], data_format='nhwc')
        
        # post-processing (draw detection)
        processed_frame = draw_detections(frame, outputs, CLASSES, conf_threshold)
        cv2.imshow('frame', processed_frame)
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # quit
    rknn_lite.release()
    cap.stop()
    cv2.destroyAllWindows()