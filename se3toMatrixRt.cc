/// \author Ankur Handa
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>

using namespace tensorflow;

REGISTER_OP("SE3toMatrixRt")
  .Input("se3_vector_input: float")
  .Output("se3_transform: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    c->set_output(0, c->MakeShape({c->Dim(input_shape,0), 3, 4}));
    return Status::OK();
  });

/// \brief Implementation of an inner product operation.
/// \param context
/// \author Ankur Handa
class SE3toMatrixRtOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit SE3toMatrixRtOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(1, context->num_inputs());
    
    // get the input tensor
    const Tensor& se3_vector_input = context->input(0);
    
    // check shapes of input and weights
    const TensorShape& se3_vector_shape = se3_vector_input.shape();
    
    // check input is a standing vector
    DCHECK_EQ(se3_vector_input.dims(), 2);
    DCHECK_EQ(se3_vector_input.dim_size(1), 6);
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(se3_vector_shape.dim_size(0));
    output_shape.AddDim(3);
    output_shape.AddDim(4);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto se3_vector_input_tensor = se3_vector_input.tensor<float, 2>();
    auto output_tensor = output->tensor<float,3>();


    for (int b = 0; b < output->shape().dim_size(0); b++)
    {
        auto omega_x = se3_vector_input_tensor(b,0);
        auto omega_y = se3_vector_input_tensor(b,1);
        auto omega_z = se3_vector_input_tensor(b,2);

        auto theta = sqrt(omega_x*omega_x + omega_y * omega_y + omega_z * omega_z);

        auto sin_term = sin(theta)/theta;
        auto cos_term = (1 - cos(theta))/(theta*theta);

        if (theta < 1e-12)
        {
            cos_term = 0;
            sin_term = 1;
        }

        output_tensor(b,0,0) = 1 + cos_term * (-omega_z*omega_z - omega_y*omega_y);
        output_tensor(b,0,1) = -sin_term * omega_z + cos_term * omega_x * omega_y;
        output_tensor(b,0,2) =  sin_term * omega_y + cos_term * omega_x * omega_z;

        output_tensor(b,1,0) = sin_term * omega_z + cos_term * omega_x * omega_y;
        output_tensor(b,1,1) = 1 + cos_term * (-omega_z * omega_z - omega_x * omega_x);
        output_tensor(b,1,2) = -sin_term * omega_x + cos_term * omega_z * omega_y;

        output_tensor(b,2,0) = -sin_term * omega_y + cos_term * omega_x * omega_z;
        output_tensor(b,2,1) = sin_term * omega_x + cos_term * omega_y * omega_z ;
        output_tensor(b,2,2) = 1 + cos_term * (-omega_x*omega_x - omega_y*omega_y);

        output_tensor(b,0,3) = se3_vector_input_tensor(b,3);
        output_tensor(b,1,3) = se3_vector_input_tensor(b,4);
        output_tensor(b,2,3) = se3_vector_input_tensor(b,5);
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("SE3toMatrixRt").Device(DEVICE_CPU), SE3toMatrixRtOp);
