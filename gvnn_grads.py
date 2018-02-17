import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
gvnn_grad_modules = tf.load_op_library('build/libgvnn_grad.so')

@ops.RegisterGradient("SE3Transform")
def se3_transform_grad_cc(op, grad):
    return gvnn_grad_modules.se3_transform_points_grad(grad, op.inputs[0], op.inputs[1])
