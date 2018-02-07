/// \author Ankur Handa
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <sophus/se3.hpp>

using namespace tensorflow;

REGISTER_OP("SE3Transform")
  .Input("input: float")
  .Input("transformations: float")
  .Output("transformed_points: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));

    shape_inference::ShapeHandle se3_transform_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &se3_transform_shape));
    
    /*shape_inference::DimensionHandle output_rows = c->Dim(input_shape, 0);
  
    shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle weight_cols = c->Dim(weight_shape, 1);
    shape_inference::DimensionHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));
    */

    c->set_output(0, c->input(0));
    return Status::OK();
  });

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class SE3TransformOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit SE3TransformOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(2, context->num_inputs());
    
    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& se3_transform = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& se3_transform_shape = se3_transform.shape();
    
    // check input is a standing vector
    DCHECK_EQ(input_shape.dims(), 3);
    DCHECK_EQ(input_shape.dim_size(2), 3);
    
    // check weights is matrix of correct size
    DCHECK_EQ(se3_transform_shape.dims(), 3);
    //DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    DCHECK_EQ(se3_transform_shape.dim_size(1), 3);
    DCHECK_EQ(se3_transform_shape.dim_size(2), 4);
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));
    output_shape.AddDim(input_shape.dim_size(1));
    output_shape.AddDim(input_shape.dim_size(2));
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.tensor<float, 3>();
    auto se3_transform_tensor = se3_transform.tensor<float,3>();
    auto output_tensor = output->tensor<float,3>();

    auto r00 = se3_transform_tensor(0,0,0);
    auto r01 = se3_transform_tensor(0,0,1);
    auto r02 = se3_transform_tensor(0,0,2);
    auto t0  = se3_transform_tensor(0,0,3);	

    auto r10 = se3_transform_tensor(0,1,0);
    auto r11 = se3_transform_tensor(0,1,1);
    auto r12 = se3_transform_tensor(0,1,2);
    auto t1  = se3_transform_tensor(0,1,3);	

    auto r20 = se3_transform_tensor(0,2,0);
    auto r21 = se3_transform_tensor(0,2,1);
    auto r22 = se3_transform_tensor(0,2,2);
    auto t2  = se3_transform_tensor(0,2,3);	

    for (int h = 0; h < output->shape().dim_size(0); h++){
      for (int w = 0; w < output->shape().dim_size(1); w++){

	auto X = input_tensor(h,w,0);
	auto Y = input_tensor(h,w,1);
	auto Z = input_tensor(h,w,2);
	

	output_tensor(h,w,0) = r00 * X + r01 * Y + r02* Z + t0;     
	output_tensor(h,w,1) = r10 * X + r11 * Y + r12* Z + t1;   
	output_tensor(h,w,2) = r20 * X + r21 * Y + r22* Z + t2;     
	}
    }

   /* for (int i = 0; i < output->shape().dim_size(0); i++) {
      output_tensor(i, 0) = 0;
      for (int j = 0; j < weights.shape().dim_size(1); j++) {
        output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
      }
    }*/
  }
};

REGISTER_KERNEL_BUILDER(Name("SE3Transform").Device(DEVICE_CPU), SE3TransformOp);
