#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>


template<
    typename T,
    int kHeadSize
>
__global__ void sdp_attention_forward_kernel(
    T* __restrict__ out_ptr,
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    const int* __restrict__ attn_mask_ptr,
    int q_stride,
    int kv_head_stride,
    int num_seqs,
    int num_heads,
    int head_size
) {
    // TODO

    //     # q * k_transposed / sqrt(dk)
    //     dk = self.head_size
    //     qk = q2.matmul(k2.transpose(-2, -1)) / math.sqrt(dk)
    //
    //     # out = softmax(qk) * v (no out projection for now)
    //     mh_attn = nn.functional.softmax(qk, dim=-1)
    //     mh_attn_out = mh_attn.matmul(v2)
    //
    //     # Rearrange data back as [seq_length, embedding_dim]
    //     mh_attn_out = mh_attn_out.transpose(1, 2).reshape(batches, seq_length, self.embedding_dim)
    //     mh_attn_out = F.linear(mh_attn_out, self.out_proj)
}


template <
    typename T,
    int kHeadSize
>
void sdp_attention_forward_dtype_hsize(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor> opt_attn_mask,
    bool is_causal) {

    // TODO Make sure same size across tensors

    // Out and QKV
    T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
    T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
    T* key_ptr = reinterpret_cast<T*>(key.data_ptr());
    T* value_ptr = reinterpret_cast<T*>(value.data_ptr());

    // Attention mask
    int* attn_mask_ptr = nullptr;
    if (opt_attn_mask.has_value()) {
        attn_mask_ptr = reinterpret_cast<int*>(opt_attn_mask.value().data_ptr());
    } else {
        // TODO build it
    }

    // [num_seqs, num_heads, head_size]
    const auto sizes = query.sizes();
    int num_seqs = sizes[0]; // flattened batches * seqs
    int num_heads = sizes[1];
    int head_size = sizes[2];
    int seq_stride = query.stride(0);
    int head_stride = query.stride(1); // TODO Should be from KV_Cache?

    const dim3 block_count(num_heads, num_seqs);
    const dim3 block_size(kHeadSize); // TODO Fix it
    const int32_t shared_mem_size = 256 * 12 * sizeof(float);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sdp_attention_forward_kernel<T, kHeadSize><<<block_count, block_size, shared_mem_size, stream>>>(
        out_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        attn_mask_ptr,
        seq_stride,
        head_stride,
        num_seqs,
        num_heads,
        head_size
        );
}


template <typename T>
void sdp_attention_forward_dtype(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor> opt_attn_mask,
    bool is_causal) {
    // TODO Handle other head sizes
    sdp_attention_forward_dtype_hsize<T, 128>(out, query, key, value, opt_attn_mask, is_causal);
}


at::Tensor sdp_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor> opt_attn_mask,
    bool is_causal) {

    const auto sizes = query.sizes();
    TORCH_CHECK(sizes.size() == 3,
        "query must be [seqs, heads, head_size]");

    auto dtype = query.dtype();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
        "query type must be fp16 or bf16");

    // Allocate output
    at::Tensor out = torch::zeros_like(query);

    if (dtype == torch::kFloat16) {
        sdp_attention_forward_dtype<__half>(out, query, key, value, opt_attn_mask, is_causal);
    } else if (dtype == torch::kBFloat16) {
        sdp_attention_forward_dtype<__nv_bfloat16>(out, query, key, value, opt_attn_mask, is_causal);
    }

    return out;
}
