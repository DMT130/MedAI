import onnx

import onnxruntime as ort
import numpy as np

sess_options = ort.SessionOptions()
# Below is for optimizing performance
sess_options.intra_op_num_threads = 24
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession("xray.onnx", sess_options=sess_options)

#dummpy_input = torch.randn(1, 3, 1024, 1024)
def onnxpredic(dummpy_input):
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: dummpy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

# out = onnxpredic(dummpy_input)

# print(out)