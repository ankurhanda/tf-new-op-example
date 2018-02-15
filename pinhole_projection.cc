/// \author Ankur Handa
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("PinholeProjection")
  .Input("depth_map: float")
  .Input("inverse_camera_matrix: float")
  .Output("points_3d: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));

    shape_inference::ShapeHandle inverse_camera_matrix_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &inverse_camera_matrix_shape));
    
    c->set_output(0, c->Tensor(c->Dim(input_shape,0), c->Dim(input_shape,1), c->Dim(input_shape,2), 3));
    return Status::OK();
  });

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class PinholeProjectionOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit PinholeProjectionOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());
    
    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& inverse_camera_matrix = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& inverse_camera_matrix_shape = inverse_camera_matrix.shape();
    
    // check input is a standing vector
    DCHECK_EQ(input_shape.dims(), 3);
    DCHECK_EQ(input_shape.dim_size(2), 3);
    
    // check weights is matrix of correct size
    DCHECK_EQ(input_shape.dims(), 3);
    //DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    DCHECK_EQ(inverse_camera_matrix_shape.dim_size(1), 3);
    DCHECK_EQ(inverse_camera_matrix_shape.dim_size(2), 3);
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));
    output_shape.AddDim(input_shape.dim_size(1));
    output_shape.AddDim(input_shape.dim_size(2));
    output_shape.AddDim(3);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.tensor<float, 3>();
    auto inverse_camera_matrix_tensor = inverse_camera_matrix.tensor<float,3>();
    auto output_tensor = output->tensor<float,4>();

    auto K00 = input_camera_matrix_tensor(0,0,0);
    auto K02 = input_camera_matrix_tensor(0,0,2);
    auto K11 = input_camera_matrix_tensor(0,1,1);
    auto K12 = input_camera_matrix_tensor(0,1,2);	

    for (int h = 0; h < output->shape().dim_size(0); h++){
      for (int w = 0; w < output->shape().dim_size(1); w++){

	auto X = w;
	auto Y = h;
	auto depth = input_tensor(h,w);
	
	output_tensor(h,w,0) = (K00 * X + K02)*depth;     
	output_tensor(h,w,1) = (K11 * Y + K12)*depth;   
	output_tensor(h,w,2) = depth;     
	}
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("SE3Transform").Device(DEVICE_CPU), SE3TransformOp);
