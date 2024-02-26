//#include <torch/extension.h>
#include <torch/python.h>
#include <torch/nn/functional.h>

#ifndef DPRINTF
#define DPRINTF(STR, ...) { printf(STR, __VA_ARGS__); }
#endif


at::Tensor sdp_attention_forward1(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor> attn_mask, bool is_causal) {

    TORCH_CHECK(query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16,
        "data type must be fp16 or bf16");

    return torch::zeros_like(query);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MHA API";
    m.def("sdp_attention_forward", &sdp_attention_forward1, "Multi-Head Scalar Dot Product Attention Forward");
}
