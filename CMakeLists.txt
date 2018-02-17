cmake_minimum_required(VERSION 2.8)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

message("Include libraries are: ${Tensorflow_INCLUDE_DIRS}")

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

link_directories(/home/ahanda/anaconda3/lib/python3.6/site-packages/tensorflow)
#link_directories(/usr/local/lib/python3.6/site-packages/tensorflow/)
link_libraries(-ltensorflow_framework)

# build the actual operation which can be used directory
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(gvnn SHARED inner_product.cc se3_transform.cc se3toMatrixRt.cc)

# build the gradient operation which is used in inner_product_grad.py
# to register it
include_directories(${Tensorflow_INCLUDE_DIRS})
add_library(gvnn_grad SHARED inner_product_grad.cc se3_transform_points_grad.cc se3toMatrixRt_grad.cc)
