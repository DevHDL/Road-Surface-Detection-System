## Inference System

### System Information

| 항목    | 내용                              |
|--------|-----------------------------------|
| Board  | Orange Pi 5                       |
| OS     | Armbian 25.2.1 (Bookworm)         |
| Python | 3.10.10                           |
| Model  | YOLOv5n                           |

### Set python version

```bash
pyenv install 3.10.10
pyenv virtualenv 3.10.10 road-system
pyenv local road-system
```

### Install rknn-toolkit-lite2

[[Github Repo] rknn-toolkit-lite2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages)

```bash
pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### Execute python

```bash
(road-system) python inference.py
```

### inference.py

#### Load RKNN Model
```python
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
```

#### Init CV
```python
# init video stream
vs = cv2.VideoCapture('assets/original_video.mp4')
vs.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

if not vs.isOpened():
    print('Cannot capture from camera. Exiting.')
    quit()
```

#### Model Output

- model inference
```python
outputs = rknn_lite.inference(inputs=[frame], data_format='nhwc')
```

- output format
```
(1, 1, 22500, 9)
```

- 해석
```
[center_x, center_y, width, height, Confidence, class0_확률, class1_확률, ..., class3_확률]
```

- Detection
```python
for detection in 22500:
    if confidence > conf_threshold:
        x_min, y_min, x_max, y_max = get_bbox_coordinates(detection[:4])
        class_name = get_top_score_object(detection[5:])
```