#include <torch/extension.h>
#include <torch/python.h>

#ifndef DPRINTF
#define DPRINTF(STR, ...) { printf(STR, __VA_ARGS__); }
#endif

at::Tensor sdp_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor> attn_mask,
    bool is_causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MHA API";
    m.def("sdp_attention_forward", &sdp_attention_forward, "Multi-Head Scalar Dot Product Attention Forward");
}
