#include <torch/extension.h>

at::Tensor conv2d_fwd(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias) {
    namespace F = torch::nn::functional;
    at::Tensor output = F::conv2d(input, weight, F::Conv2dFuncOptions().bias(bias).stride(1).padding(0));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NN API";
    m.def("conv2d_fwd", &conv2d_fwd, "Conv2d Forward");
}
