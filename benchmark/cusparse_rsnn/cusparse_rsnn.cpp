#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

void launch_lif_step(
    const float* input,
    float* v,
    const float* psc,
    float* spikes,
    int count,
    float dt,
    float tau_mem,
    float v_threshold,
    float v_reset,
    float c_m,
    cudaStream_t stream);

void launch_psc_step(
    float* psc,
    const float* recurrent,
    int count,
    float decay,
    cudaStream_t stream);

namespace {

#define CUSPARSE_CHECK(expression)                                      \
    do {                                                               \
        const cusparseStatus_t status = (expression);                   \
        TORCH_CHECK(                                                    \
            status == CUSPARSE_STATUS_SUCCESS,                         \
            "cuSPARSE failure: ",                                     \
            cusparseGetErrorString(status));                           \
    } while (false)

struct Plan {
    int n_neuron;
    int batch_size;
    int t_steps;
    bool use_spmv;
    cusparseSpMatDescr_t matrix = nullptr;
    std::vector<cusparseDnVecDescr_t> spike_vectors;
    cusparseDnVecDescr_t recurrent_vector = nullptr;
    std::vector<cusparseDnMatDescr_t> spike_matrices;
    cusparseDnMatDescr_t recurrent_matrix = nullptr;
    torch::Tensor crow;
    torch::Tensor col;
    torch::Tensor weight;
    torch::Tensor spikes;
    torch::Tensor recurrent;
    torch::Tensor workspace;
};

std::mutex plan_mutex;
std::unordered_map<int64_t, std::shared_ptr<Plan>> plans;
int64_t next_plan_id = 1;

void validate_cuda_float(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32.");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

int64_t prepare(
    torch::Tensor crow,
    torch::Tensor col,
    torch::Tensor weight,
    torch::Tensor spikes,
    torch::Tensor recurrent,
    int64_t batch_size) {
    c10::cuda::CUDAGuard guard(weight.device());
    TORCH_CHECK(crow.is_cuda() && col.is_cuda(), "CSR indices must be CUDA tensors.");
    TORCH_CHECK(crow.scalar_type() == torch::kInt32, "crow must be int32.");
    TORCH_CHECK(col.scalar_type() == torch::kInt32, "col must be int32.");
    TORCH_CHECK(
        crow.is_contiguous() && col.is_contiguous(),
        "CSR indices must be contiguous.");
    validate_cuda_float(weight, "weight");
    validate_cuda_float(spikes, "spikes");
    validate_cuda_float(recurrent, "recurrent");
    TORCH_CHECK(spikes.dim() == 3, "spikes must have shape (T, B, N).");
    TORCH_CHECK(recurrent.dim() == 2, "recurrent must have shape (B, N).");
    TORCH_CHECK(spikes.size(1) == batch_size, "spike batch size mismatch.");
    TORCH_CHECK(recurrent.size(0) == batch_size, "recurrent batch size mismatch.");
    const int64_t t_steps = spikes.size(0);
    const int64_t n_neuron = spikes.size(2);
    TORCH_CHECK(recurrent.size(1) == n_neuron, "recurrent neuron size mismatch.");
    TORCH_CHECK(crow.numel() == n_neuron + 1, "crow must have N + 1 entries.");
    TORCH_CHECK(col.numel() == weight.numel(), "column and weight counts differ.");
    TORCH_CHECK(batch_size > 0 && t_steps > 0 && n_neuron > 0, "invalid dimensions.");

    auto plan = std::make_shared<Plan>();
    plan->n_neuron = static_cast<int>(n_neuron);
    plan->batch_size = static_cast<int>(batch_size);
    plan->t_steps = static_cast<int>(t_steps);
    plan->use_spmv = batch_size == 1;
    plan->crow = crow;
    plan->col = col;
    plan->weight = weight;
    plan->spikes = spikes;
    plan->recurrent = recurrent;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &plan->matrix,
        n_neuron,
        n_neuron,
        weight.numel(),
        crow.data_ptr<int>(),
        col.data_ptr<int>(),
        weight.data_ptr<float>(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    size_t workspace_bytes = 0;
    if (plan->use_spmv) {
        plan->spike_vectors.resize(t_steps);
        for (int64_t t = 0; t < t_steps; ++t) {
            CUSPARSE_CHECK(cusparseCreateDnVec(
                &plan->spike_vectors[t],
                n_neuron,
                spikes.data_ptr<float>() + t * n_neuron,
                CUDA_R_32F));
        }
        CUSPARSE_CHECK(cusparseCreateDnVec(
            &plan->recurrent_vector,
            n_neuron,
            recurrent.data_ptr<float>(),
            CUDA_R_32F));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            plan->matrix,
            plan->spike_vectors[0],
            &beta,
            plan->recurrent_vector,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &workspace_bytes));
    } else {
        plan->spike_matrices.resize(t_steps);
        for (int64_t t = 0; t < t_steps; ++t) {
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &plan->spike_matrices[t],
                n_neuron,
                batch_size,
                n_neuron,
                spikes.data_ptr<float>() + t * batch_size * n_neuron,
                CUDA_R_32F,
                CUSPARSE_ORDER_COL));
        }
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &plan->recurrent_matrix,
            n_neuron,
            batch_size,
            n_neuron,
            recurrent.data_ptr<float>(),
            CUDA_R_32F,
            CUSPARSE_ORDER_COL));
        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            plan->matrix,
            plan->spike_matrices[0],
            &beta,
            plan->recurrent_matrix,
            CUDA_R_32F,
            CUSPARSE_SPMM_CSR_ALG1,
            &workspace_bytes));
    }
    plan->workspace = torch::empty(
        {static_cast<int64_t>(workspace_bytes)},
        weight.options().dtype(torch::kUInt8));

    std::lock_guard<std::mutex> lock(plan_mutex);
    const int64_t plan_id = next_plan_id++;
    plans.emplace(plan_id, std::move(plan));
    return plan_id;
}

std::shared_ptr<Plan> get_plan(int64_t plan_id) {
    std::lock_guard<std::mutex> lock(plan_mutex);
    const auto iterator = plans.find(plan_id);
    TORCH_CHECK(iterator != plans.end(), "Unknown direct cuSPARSE plan.");
    return iterator->second;
}

int64_t get_workspace_bytes(int64_t plan_id) {
    auto plan = get_plan(plan_id);
    return plan->workspace.numel() * plan->workspace.element_size();
}

void run(
    int64_t plan_id,
    torch::Tensor input,
    torch::Tensor v,
    torch::Tensor psc,
    double dt,
    double tau_mem,
    double tau_syn,
    double v_threshold,
    double v_reset,
    double c_m) {
    auto plan = get_plan(plan_id);
    c10::cuda::CUDAGuard guard(input.device());
    validate_cuda_float(input, "input");
    validate_cuda_float(v, "v");
    validate_cuda_float(psc, "psc");
    TORCH_CHECK(
        input.sizes() == plan->spikes.sizes(),
        "input must have the prepared (T, B, N) shape.");
    TORCH_CHECK(v.sizes() == psc.sizes(), "v and psc shapes differ.");
    TORCH_CHECK(v.numel() == plan->batch_size * plan->n_neuron, "state size mismatch.");
    TORCH_CHECK(
        dt > 0.0 && tau_mem > 0.0 && tau_syn > 0.0 && c_m > 0.0,
        "invalid parameters.");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
    CUSPARSE_CHECK(cusparseSetStream(handle, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(v.data_ptr<float>(), 0, v.nbytes(), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(psc.data_ptr<float>(), 0, psc.nbytes(), stream));

    const int count = plan->batch_size * plan->n_neuron;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float decay = std::exp(-static_cast<float>(dt / tau_syn));
    void* workspace = plan->workspace.numel() == 0
        ? nullptr
        : plan->workspace.data_ptr();
    for (int t = 0; t < plan->t_steps; ++t) {
        float* spike_t =
            plan->spikes.data_ptr<float>() + static_cast<int64_t>(t) * count;
        const float* input_t =
            input.data_ptr<float>() + static_cast<int64_t>(t) * count;
        launch_lif_step(
            input_t,
            v.data_ptr<float>(),
            psc.data_ptr<float>(),
            spike_t,
            count,
            static_cast<float>(dt),
            static_cast<float>(tau_mem),
            static_cast<float>(v_threshold),
            static_cast<float>(v_reset),
            static_cast<float>(c_m),
            stream);
        if (plan->use_spmv) {
            CUSPARSE_CHECK(cusparseSpMV(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                plan->matrix,
                plan->spike_vectors[t],
                &beta,
                plan->recurrent_vector,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                workspace));
        } else {
            CUSPARSE_CHECK(cusparseSpMM(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                plan->matrix,
                plan->spike_matrices[t],
                &beta,
                plan->recurrent_matrix,
                CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG1,
                workspace));
        }
        launch_psc_step(
            psc.data_ptr<float>(),
            plan->recurrent.data_ptr<float>(),
            count,
            decay,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prepare", &prepare, "Prepare direct cuSPARSE RSNN descriptors");
    module.def(
        "workspace_bytes",
        &get_workspace_bytes,
        "Return direct cuSPARSE workspace bytes");
    module.def("run", &run, "Run direct cuSPARSE RSNN forward");
}
