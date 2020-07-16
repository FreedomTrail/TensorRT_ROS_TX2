import os
import sys
import cv2
import time
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import coco
import uff
import tensorrt as trt
import graphsurgeon as gs
import common

#from config import model_ssd_inception_v2_coco_2017_11_17 as model 
#from config import model_ssd_mobilenet_v1_coco_2018_01_28 as model
#from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model
from config import model_ssdlite_mobilenet_v2_coco_2018_05_09 as model

#ctypes.CDLL("lib/libflattenconcat.so")
COCO_LABELS = coco.COCO_CLASSES_LIST

# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)

# step1: compile model into TensorRT
if not os.path.isfile(model.TRTbin):
    dynamic_graph = model.add_plugin(gs.DynamicGraph(model.path))
    uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename='tmp.uff')

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True
        parser.register_input('Input', model.dims)
        parser.register_output('MarkOutput_0')
        parser.parse('tmp.uff', network)
        engine = builder.build_cuda_engine(network)
        buf = engine.serialize()
        with open(model.TRTbin, 'wb') as f:
            f.write(buf)

# step2: create engine
with open(model.TRTbin, 'rb') as f:
    buf = f.read()
    engine = runtime.deserialize_cuda_engine(buf)

# step3: create buffer
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
       
# step4: create context
context = engine.create_execution_context()

# step5: input data preprocess
ori = cv2.imread(sys.argv[1])
image = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (model.dims[2],model.dims[1]))
image = (2.0/255.0) * image - 1.0
image = image.transpose((2, 0, 1))
np.copyto(inputs[0].host, image.ravel())

# step6: do inference
trt_outputs = []
start_time = time.time()
trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
results = trt_outputs[0]
print("infer times "+str(time.time() - start_time))

# step7: postprocess    
height, width, channels = ori.shape
for i in range(int(len(results)/model.layout)):
    prefix = i*model.layout
    conf  = results[prefix+2]
    if conf > 0.7:
        index = int(results[prefix+0])
        label = int(results[prefix+1])
        xmin  = int(results[prefix+3]*width)
        ymin  = int(results[prefix+4]*height)
        xmax  = int(results[prefix+5]*width)
        ymax  = int(results[prefix+6]*height)
        print("Detected {} with confidence {}".format(COCO_LABELS[label], "{0:.0%}".format(conf)))
        cv2.rectangle(ori, (xmin,ymin), (xmax, ymax), (0,0,255),3)
        cv2.putText(ori, COCO_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
print("execute times "+str(time.time() - start_time))
cv2.imwrite("result.jpg", ori)
cv2.imshow("result", ori)
cv2.waitKey(0)
