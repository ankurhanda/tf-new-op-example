/// \file inner_product_grad.cc
/// \author David Stutz
/// \brief Implementation of the gradient of a inner product operation, see
/// inner_product.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace Sophus;
using namespace Eigen;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("SE3toMatrixRtGrad")
  .Input("grad: float32")
  .Input("se3_vector: float32")
  .Output("grad_se3: float32");

namespace tensorflow{

using namespace tensorflow;

/// \brief Implementation of an inner product gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
/// \author David Stutz
class SE3toMatrixRtGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit SE3toMatrixRtGradOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  Eigen::Matrix<float,3,3> skew_symmetric(Eigen::Matrix<float,3,1>v)
  {
    Eigen::Matrix<float, 3,3>v_cross;
        v_cross << 0, -v(2,0), v(1,0),
                   v(2,0), 0, -v(0,0),
                  -v(1,0), v(0,0), 0;

    return v_cross;

  }

  /// \brief Compute the inner product gradients.
  /// \param context
  void Compute(OpKernelContext* context) override {

    // output and grad is provided as input
    DCHECK_EQ(2, context->num_inputs());

    // get the gradient tensor
    const Tensor& grad = context->input(0);

    // get the original input tensor
    const Tensor& se3_vector = context->input(1);

    // create input shape (inferred from the additional attribute `n`)
    TensorShape se3_vector_shape = se3_vector.shape();

    DCHECK_EQ(se3_vector_shape.dim_size(0), grad.shape().dim_size(0));

    // create output tensors
    Tensor* grad_se3 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, se3_vector_shape, &grad_se3));

    // get the Eigen tensors for data access
    auto grad_tensor = grad.tensor<float, 3>();
    auto se3_vector_tensor = se3_vector.tensor<float,2>();
    auto grad_se3_tensor = grad_se3->tensor<float,2>();

    for(int b = 0; b < se3_vector_shape.dim_size(0); b++)
    {
        auto v1 = se3_vector_tensor(b,0);
        auto v2 = se3_vector_tensor(b,1);
        auto v3 = se3_vector_tensor(b,2);

        auto v_mag = sqrt(v1*v1 + v2*v2 + v3*v3);

        Eigen::Matrix<float,3,1>vec;
        vec<< v1, v2, v3;

        Eigen::Matrix<float,3,3> R = SO3<float>::exp(vec).matrix();
        Eigen::Matrix<float, 3,3>v_cross = skew_symmetric(vec);
        Eigen::Matrix<float,3,3> I = Eigen::Matrix<float,3,3>::Identity(3,3);

        Eigen::Matrix<float, 3, 3>v_cross_I_minus_R = v_cross  * (I-R);

        Eigen::Matrix<float,3,1> e_i;
        e_i << 1, 0, 0;

        Eigen::Matrix<float, 3, 3>dR_dv1 = v1 * v_cross;
        Eigen::Matrix<float, 3, 3>v_cross_I_minus_R_times_e1 = skew_symmetric(v_cross_I_minus_R * e_i);

        dR_dv1 += v_cross_I_minus_R_times_e1;
        dR_dv1 *= R;
        dR_dv1 /= v_mag;

        e_i << 0, 1, 0;

        Eigen::Matrix<float, 3, 3>dR_dv2 = v2 * v_cross;
        Eigen::Matrix<float, 3, 3>v_cross_I_minus_R_times_e2 = skew_symmetric(v_cross_I_minus_R * e_i);

        dR_dv2 += v_cross_I_minus_R_times_e2;
        dR_dv2 *= R;
        dR_dv2 /= v_mag;

        e_i << 0, 0, 1;

        Eigen::Matrix<float, 3, 3>dR_dv3 = v3 * v_cross;
        Eigen::Matrix<float, 3, 3>v_cross_I_minus_R_times_e3 = skew_symmetric(v_cross_I_minus_R * e_i);

        dR_dv3 += v_cross_I_minus_R_times_e3;
        dR_dv3 *= R;
        dR_dv3 /= v_mag;

    }

  }
};

REGISTER_KERNEL_BUILDER(Name("SE3toMatrixRtGrad").Device(DEVICE_CPU), SE3toMatrixRtGradOp);
};