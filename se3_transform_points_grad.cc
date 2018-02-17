/// \file inner_product_grad.cc
/// \author David Stutz
/// \brief Implementation of the gradient of a inner product operation, see
/// inner_product.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("SE3TransformPointsGrad")
  .Input("grad: float32")
  .Input("input_points: float32")
  .Input("se3_transforms: float32")
  .Output("grad_input_points: float32")
  .Output("grad_se3_transforms: float32");

/// \brief Implementation of an inner product gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
/// \author David Stutz
class SE3TransformPointsGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit SE3TransformPointsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product gradients.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // output and grad is provided as input
    DCHECK_EQ(3, context->num_inputs());

    // get the gradient tensor
    const Tensor& grad = context->input(0);
    
    // get the original input tensor
    const Tensor& input_points = context->input(1);
    
    // get the weight tensor
    const Tensor& se3_transforms = context->input(2);
    
    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_points_shape = input_points.shape();
    TensorShape se3_transforms_shape = se3_transforms.shape();
    
    DCHECK_EQ(input_points_shape.dim_size(0), se3_transforms_shape.dim_size(0));
    DCHECK_EQ(se3_transforms_shape.dim_size(0), grad.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_input_points = NULL;
    Tensor* grad_se3_transforms = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_points_shape, &grad_input_points));
    OP_REQUIRES_OK(context, context->allocate_output(1, se3_transforms_shape, &grad_se3_transforms));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.tensor<float, 4>();
    auto se3_transforms_tensor = se3_transforms.tensor<float,3>();
    auto input_points_tensor = input_points.tensor<float,4>();
    auto grad_input_points_tensor = grad_input_points->tensor<float,4>();
    auto grad_se3_transforms_tensor = grad_se3_transforms->tensor<float,3>();
   
   
    for (int b = 0; b < input_points_shape.dim_size(0); b++)
    {
	auto r_00 = se3_transforms_tensor(b,0,0);
	auto r_01 = se3_transforms_tensor(b,0,1);
	auto r_02 = se3_transforms_tensor(b,0,2);
	auto  t_0 = se3_transforms_tensor(b,0,3);
	
	auto r_10 = se3_transforms_tensor(b,1,0);
	auto r_11 = se3_transforms_tensor(b,1,1);
	auto r_12 = se3_transforms_tensor(b,1,2);
	auto t_1 = se3_transforms_tensor(b,1,3);
	
	auto r_20 = se3_transforms_tensor(b,2,0);
	auto r_21 = se3_transforms_tensor(b,2,1);
	auto r_22 = se3_transforms_tensor(b,2,2);
	auto t_2 = se3_transforms_tensor(b,2,3);
	
	for(int h=0; h < input_points_shape.dim_size(1); h++)
	{
	   for(int w = 0; w < input_points_shape.dim_size(2); w++)
	   {
		grad_input_points_tensor(b,h,w,0) = grad_tensor(b,h,w,0)*r_00 + grad_tensor(b,h,w,1)*r_10 + grad_tensor(b,h,w,2) * r_20;
		grad_input_points_tensor(b,h,w,1) = grad_tensor(b,h,w,0)*r_01 + grad_tensor(b,h,w,1)*r_11 + grad_tensor(b,h,w,2) * r_21;
		grad_input_points_tensor(b,h,w,2) = grad_tensor(b,h,w,0)*r_02 + grad_tensor(b,h,w,1)*r_12 + grad_tensor(b,h,w,2) * r_22;
	   }
	}

    }
    

    for (int b =0; b < input_points_shape.dim_size(0); b++)
    {

	auto r_00 = se3_transforms_tensor(b,0,0);
	auto r_01 = se3_transforms_tensor(b,0,1);
	auto r_02 = se3_transforms_tensor(b,0,2);
	auto  t_0 = se3_transforms_tensor(b,0,3);
	
	auto r_10 = se3_transforms_tensor(b,1,0);
	auto r_11 = se3_transforms_tensor(b,1,1);
	auto r_12 = se3_transforms_tensor(b,1,2);
	auto t_1 = se3_transforms_tensor(b,1,3);
	
	auto r_20 = se3_transforms_tensor(b,2,0);
	auto r_21 = se3_transforms_tensor(b,2,1);
	auto r_22 = se3_transforms_tensor(b,2,2);
	auto t_2 = se3_transforms_tensor(b,2,3);

	grad_se3_transforms_tensor(b,0,0)=0;
	grad_se3_transforms_tensor(b,0,1)=0;
	grad_se3_transforms_tensor(b,0,2)=0;
	grad_se3_transforms_tensor(b,0,3)=0;

	grad_se3_transforms_tensor(b,1,0)=0;
	grad_se3_transforms_tensor(b,1,1)=0;
	grad_se3_transforms_tensor(b,1,2)=0;
	grad_se3_transforms_tensor(b,1,3)=0;

	grad_se3_transforms_tensor(b,2,0)=0;
	grad_se3_transforms_tensor(b,2,1)=0;
	grad_se3_transforms_tensor(b,2,2)=0;
	grad_se3_transforms_tensor(b,2,3)=0;
	
	for(int h = 0; h < input_points_shape.dim_size(1); h++)
	{
	   for(int w = 0; w < input_points_shape.dim_size(2); w++)
	   {
		grad_se3_transforms_tensor(b,0,0) += r_00 * grad_tensor(b,h,w,0); 
		grad_se3_transforms_tensor(b,0,1) += r_01 * grad_tensor(b,h,w,0); 
		grad_se3_transforms_tensor(b,0,2) += r_02 * grad_tensor(b,h,w,0); 
		grad_se3_transforms_tensor(b,0,3) +=  t_0 * grad_tensor(b,h,w,0); 
		
		grad_se3_transforms_tensor(b,1,0) += r_10 * grad_tensor(b,h,w,1); 
		grad_se3_transforms_tensor(b,1,1) += r_11 * grad_tensor(b,h,w,1); 
		grad_se3_transforms_tensor(b,1,2) += r_12 * grad_tensor(b,h,w,1); 
		grad_se3_transforms_tensor(b,1,3) +=  t_1 * grad_tensor(b,h,w,1); 
		
		grad_se3_transforms_tensor(b,2,0) += r_20 * grad_tensor(b,h,w,2); 
		grad_se3_transforms_tensor(b,2,1) += r_21 * grad_tensor(b,h,w,2); 
		grad_se3_transforms_tensor(b,2,2) += r_22 * grad_tensor(b,h,w,2); 
		grad_se3_transforms_tensor(b,2,3) +=  t_2 * grad_tensor(b,h,w,2); 
			
	   }
	}
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("SE3TransformPointsGrad").Device(DEVICE_CPU), SE3TransformPointsGradOp);
