from re import L
import onnx
import onnxruntime
import onnx_tf


model = onnx.load("/Users/jperrsau/cu-src/thesis/src/nnenum/examples/acasxu/data/ACASXU_run2a_1_1_batch_2000.onnx")


for node in model.graph.node:
    print(node)

#tf_model = onnx_tf.backend.prepare(model)


# import  tensorflow as tf

# model = tf.saved_model.load("./data/acasxu/ACASXU_run2a_1_1_batch_2000.pb")

# print(model)
