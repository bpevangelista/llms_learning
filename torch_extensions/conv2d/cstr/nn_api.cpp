#include <torch/extension.h>

#ifndef DPRINTF
#define DPRINTF(STR, ...) { printf(STR, __VA_ARGS__); }
#endif

at::Tensor conv2d_cpp_fwd(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias) {
    TORCH_CHECK(input.device().type() == torch::kCPU);
    TORCH_CHECK(weight.device().type() == torch::kCPU);
    TORCH_CHECK(bias.device().type() == torch::kCPU);

    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only support Float32");
    TORCH_CHECK(input.dtype() == weight.dtype(), "Input and weight must have same dtype");
    TORCH_CHECK(input.dtype() == bias.dtype(), "Input and bias must have same dtype");

    auto input_shape = input.sizes();
    TORCH_CHECK(input_shape.size() == 4, "Input must have N, C, H, W shape");
    int64_t batches = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    DPRINTF("INPUT %d %d %d %d \n", batches, channels, height, width);

    auto weight_shape = weight.sizes();
    TORCH_CHECK(weight_shape.size() == 4, "Weight must have OUT_C, IN_C, Kh, Kw shape");
    int64_t kernel_out_c = weight_shape[0];
    int64_t kernel_in_c = weight_shape[1];
    int64_t kernel_height = weight_shape[2];
    int64_t kernel_width = weight_shape[3];
    TORCH_CHECK(channels == kernel_in_c, "Input and Weight must have same input channels count");
    DPRINTF("WEIGHT %d %d %d %d \n", kernel_out_c, kernel_in_c, kernel_height, kernel_width);

    // TODO Fixed Stride = 1, Padding = 0
    // TODO Center sampling for even kernel size
    int32_t skip_h = kernel_height / 2;
    int32_t skip_w = kernel_width / 2;
    int32_t kernel_hend = kernel_height - skip_h;
    int32_t kernel_wend = kernel_width - skip_w;

    // Output tensor
    int32_t stride = 1;
    int32_t padding = 0;
    int32_t output_height = (height - kernel_height + 2 * padding) / stride + 1;
    int32_t output_width = (width - kernel_width + 2 * padding) / stride + 1;
    auto output = torch::zeros({batches, kernel_out_c, output_height, output_width},
        torch::TensorOptions().dtype(torch::kFloat32));
    DPRINTF("OUTPUT %d %d %d %d \n", output.sizes()[0], output.sizes()[1], output.sizes()[2], output.sizes()[3]);

    auto input_data = input.accessor<float, 4>();
    auto weight_data = weight.accessor<float, 4>();
    auto out_data = output.accessor<float, 4>();
    //float* input_data = reinterpret_cast<float*>(input.data_ptr());
    //float* weight_data = reinterpret_cast<float*>(weight.data_ptr());
    //float* out_data = reinterpret_cast<float*>(output.data_ptr());

    // Vanilla - lets assume one batch
    for (int64_t out_c = 0; out_c < kernel_out_c; ++out_c) {
        for (int64_t in_c = 0; in_c < channels; ++in_c) {
            for (int64_t h = skip_h; h < (height - skip_h); ++h) {
                for (int64_t w = skip_w; w < (width - skip_w); ++w) {

                    // For each output pixel, accumulate over kernel
                    for (int32_t kh = -skip_h; kh < kernel_hend; ++kh) {
                        for (int32_t kw = -skip_w; kw < kernel_wend; ++kw) {
                            int samp_h = h + kh;
                            int samp_w = w + kw;

                            out_data[0][out_c][h-skip_h][w-skip_w] +=
                                input_data[0][in_c][samp_h][samp_w] *
                                weight_data[out_c][in_c][kh+skip_h][kw+skip_w];
                        }
                    }
                }
            }

            // TODO apply bias
        }
    }

    return output;
}

at::Tensor conv2d_aten_fwd(const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias) {
    namespace F = torch::nn::functional;
    at::Tensor output = F::conv2d(input, weight, F::Conv2dFuncOptions().bias(bias).stride(1).padding(0));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NN API";
    m.def("conv2d_aten_fwd", &conv2d_aten_fwd, "Conv2d ATen Forward");
    m.def("conv2d_cpp_fwd", &conv2d_cpp_fwd, "Conv2d CPU_Slow Forward");
}
