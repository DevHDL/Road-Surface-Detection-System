import os
from rknn.api import RKNN

ONNX_MODEL = 'yolov5.onnx'
RKNN_MODEL = 'yolov5.rknn'
QUANTIZE_ON = False
#DATASET = './datasets/yolo_subset_09.txt'

if __name__ == '__main__':

    # RKNN init
    rknn = RKNN(verbose=True, verbose_file='./model_build.log')

    # Config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='RK3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # RKNN release
    rknn.release()