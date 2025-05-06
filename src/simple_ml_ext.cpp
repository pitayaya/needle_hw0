#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    // 分配临时数组用于存储 logits、概率和梯度
    float *Z = new float[batch * k];          // Logits: (batch_size, num_classes)
    float *exp_Z = new float[batch * k];      // 指数化 logits
    float *probs = new float[batch * k];      // Softmax 概率
    float *grad_theta = new float[n * k];     // 参数 theta 的梯度

    // 按小批量遍历数据
    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);  // 当前批次的实际大小

        // 第一步：计算 logits Z = X_batch @ theta
        for (size_t b = 0; b < current_batch_size; ++b) {
            for (size_t c = 0; c < k; ++c) {
                Z[b * k + c] = 0.0f;
                for (size_t d = 0; d < n; ++d) {
                    Z[b * k + c] += X[(i + b) * n + d] * theta[d * k + c];
                }
            }
        }

        // 第二步：计算 exp(Z) 和 sum(exp(Z))，并对每个样本进行归一化得到概率
        for (size_t b = 0; b < current_batch_size; ++b) {
            float sum_exp = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                exp_Z[b * k + c] = std::exp(Z[b * k + c]);
                sum_exp += exp_Z[b * k + c];
            }
            // 归一化得到概率分布
            for (size_t c = 0; c < k; ++c) {
                probs[b * k + c] = exp_Z[b * k + c] / sum_exp;
            }
        }

        // 第三步：计算损失函数相对于 theta 的梯度
        std::fill(grad_theta, grad_theta + n * k, 0.0f);  // 初始化梯度为 0
        for (size_t b = 0; b < current_batch_size; ++b) {
            unsigned char true_class = y[i + b];  // 当前样本的真实类别
            for (size_t c = 0; c < k; ++c) {
                // 如果是真实类别，梯度因子为 (P - 1)，否则为 P
                float grad_factor = (c == true_class) ? (probs[b * k + c] - 1.0f) : probs[b * k + c];
                for (size_t d = 0; d < n; ++d) {
                    grad_theta[d * k + c] += X[(i + b) * n + d] * grad_factor;
                }
            }
        }

        // 第四步：使用梯度下降更新 theta
        for (size_t d = 0; d < n; ++d) {
            for (size_t c = 0; c < k; ++c) {
                theta[d * k + c] -= lr * grad_theta[d * k + c] / current_batch_size;
            }
        }
    }

    // 释放动态分配的内存
    delete[] Z;
    delete[] exp_Z;
    delete[] probs;
    delete[] grad_theta;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
