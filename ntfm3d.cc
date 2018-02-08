/// \author Ankur Handa
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("NonRigidBlendingTransform")
  .Input("input: float")
  .Input("masks: float")
  .Input("n_transformations: float")
  .Output("transformed_points: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    shape_inference::ShapeHandle masks_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &masks_shape));
    
    shape_inference::ShapeHandle n_transformations_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &n_transformations_shape));
    
    c->set_output(0, c->input(0));
    return Status::OK();
  });

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class NonRigidBlendingTransformOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit NonRigidBlendingTransformOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(3, context->num_inputs());
    
    // get the input tensor
    const Tensor& input_3d_points = context->input(0);
    
    // get the masks
    const Tensor& masks = context->input(1);
   
    // get the n transformations
    const Tensor& n_transformations = context->input(2);	

    // check shapes of input and weights
    const TensorShape& input_3d_points_shape = input.shape();
    const TensorShape& n_transformations_shape = n_transformations.shape();
    const TensorShape& masks_shape = masks.shape();

    // check input is of the format B x H x W x 3
    DCHECK_EQ(input_shape.dims(), 4);

    // check input last dim is of size 3
    DCHECK_EQ(input_shape.dim_size(3), 3);
    
    // check weights is matrix of correct size
    DCHECK_EQ(n_transformations_shape.dims(), 4);
    DCHECK_EQ(n_transformations_shape.dim_size(2), 3);
    DCHECK_EQ(n_transformations_shape.dim_size(3), 4);
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));
    output_shape.AddDim(input_shape.dim_size(1));
    output_shape.AddDim(input_shape.dim_size(2));
    output_shape.AddDim(input_shape.dim_size(3));
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_3d_points_tensor = input_3d_points.tensor<float, 4>();
    auto n_transformations_tensor = n_transformations.tensor<float,4>();
    auto masks_tensor = masks.tensor<float,4>();

    auto output_tensor = output->tensor<float,4>();

    for (int b = 0; b < output->shape().dim_size(0); b++)
    {
       for (int h = 0; h < output->shape().dim_size(1); h++)
       {
	   for(int w = 0; w < output->shape().dim_size(2); w++)
	   {
	      auto X = input_3d_points_tensor(b,h,w,0);
	      auto Y = input_3d_points_tensor(b,h,w,1);
	      auto Z = input_3d_points_tensor(b,h,w,2);

	      auto tX = 0;
	      auto tY = 0;
	      auto tZ = 0;

	      for(int k = 0; k < masks->shape().dim_size(3); k++)
	      {
		 auto r00 = n_transformations_tensor(b,0,0,k);
     		 auto r01 = n_transformations_tensor(b,0,1,k);
    		 auto r02 = n_transformations_tensor(b,0,2,k);
    		 auto t0  = n_transformations_tensor(b,0,3,k);

    		 auto r10 = n_transformations_tensor(b,1,0,k);
    		 auto r11 = n_transformations_tensor(b,1,1,k);
    		 auto r12 = n_transformations_tensor(b,1,2,k);
    		 auto t1  = n_transformations_tensor(b,1,3,k);

    		 auto r20 = n_transformations_tensor(b,2,0,k);
    		 auto r21 = n_transformations_tensor(b,2,1,k);
    		 auto r22 = n_transformations_tensor(b,2,2,k);
    		 auto t2  = n_transformations_tensor(b,2,3,k);

		 tX += mask_tensor(b,h,w,k)*(X*r00 + Y*r01 + Z*r02 + t0);		 	 
		 tY += mask_tensor(b,h,w,k)*(X*r10 + Y*r11 + Z*r12 + t1);		 	 
		 tZ += mask_tensor(b,h,w,k)*(X*r20 + Y*r21 + Z*r22 + t2);		 	 
	      }

	      output_tensor(b,h,w,0) = tX;
	      output_tensor(b,h,w,1) = tY;
	      output_tensor(b,h,w,2) = tZ;
	   }
	}
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("SE3Transform").Device(DEVICE_CPU), SE3TransformOp);
