TensorRT for Tensorflow Object Detection
======================================

## Install dependencies
$ sudo apt-get install python3-pip libhdf5-serial-dev hdf5-tools
$ pip3 install numpy pycuda --user

## Supported models:
- ssd_inception_v2_coco_2017_11_17
- ssd_mobilenet_v1_coco
- ssd_mobilenet_v2_coco                #On Jetson TX2:Inference time:18ms. 

We will keep adding new model into our supported list.

## Update graphsurgeon converter
Edit /usr/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py
```
diff --git a/node_manipulation.py b/node_manipulation.py
index d2d012a..1ef30a0 100644
--- a/node_manipulation.py
+++ b/node_manipulation.py
@@ -30,6 +30,7 @@ def create_node(name, op=None, _do_suffix=False, **kwargs):
     node = NodeDef()
     node.name = name
     node.op = op if op else name
+    node.attr["dtype"].type = 1
     for key, val in kwargs.items():
         if key == "dtype":
             node.attr["dtype"].type = val.as_datatype_enum

## RUN
**1. Maximize the TX2 performance**
```
$ sudo nvpmodel -m 0
$ sudo jetson_clocks
```
**2. Update main.py based on the model you used**
```
from config import model_ssd_inception_v2_coco_2017_11_17 as model
from config import model_ssd_mobilenet_v1_coco_2018_01_28 as model
from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model
```

**3. Execute**
```
$ python3 main.py [image]
```

Notice:
It takes some time to compile a TensorRT model when the first launching.
After that, TensorRT engine can be created directly with the serialized .bin file
To get more memory, it's recommended to turn-off X-server.

More information,Please go to https://github.com/AastaNV/TRT_object_detection, It implement on Jetson Nano.
