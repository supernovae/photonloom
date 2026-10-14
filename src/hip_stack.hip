// HIPRTC supplies the kernel qualifiers, thread indices, and device math builtins.
// All stack arrays are frame-major; one lane owns one pixel column of scratch.
__device__ float stack_median(float* scratch, unsigned int p, unsigned int np,
                              unsigned int count) {
    const unsigned int half = count / 2;
    unsigned int left = 0;
    unsigned int right = count - 1;
    while (left < right) {
        const float pivot = scratch[((left + right) / 2) * np + p];
        unsigned int lt = left;
        unsigned int i = left;
        unsigned int gt = right + 1;
        while (i < gt) {
            const float v = scratch[i * np + p];
            if (v < pivot) {
                const float old = scratch[lt * np + p];
                scratch[lt * np + p] = v;
                scratch[i * np + p] = old;
                ++lt;
                ++i;
            } else if (v > pivot) {
                --gt;
                const float old = scratch[gt * np + p];
                scratch[gt * np + p] = v;
                scratch[i * np + p] = old;
            } else {
                ++i;
            }
        }
        if (half < lt) {
            right = lt - 1;
        } else if (half >= gt) {
            left = gt;
        } else {
            break;
        }
    }
    const float upper = scratch[half * np + p];
    if (count & 1) return upper;
    float lower = scratch[p];
    for (unsigned int k = 1; k < half; ++k) {
        const float v = scratch[k * np + p];
        lower = v > lower ? v : lower;
    }
    return 0.5f * (lower + upper);
}

__device__ float stack_clamp(float v, float low, float high) {
    return v < low ? low : (v > high ? high : v);
}
__device__ float stack_max(float a, float b) { return a > b ? a : b; }

extern "C" __global__ void stack_tile(const float* values, const unsigned int* masks,
    const float* weights, float* scratch, float* output, unsigned int* output_masks,
    unsigned int np, unsigned int nf) {
    const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;
    unsigned int m = 0;
    float first = 0.0f, second = 0.0f, first_w = 0.0f, second_w = 0.0f;
    for (unsigned int k = 0; k < nf; ++k) {
        const unsigned int idx = k * np + p;
        if (masks[idx]) {
            const float v = values[idx];
            scratch[m * np + p] = v;
            if (m == 0) { first = v; first_w = weights[k]; }
            else if (m == 1) { second = v; second_w = weights[k]; }
            ++m;
        }
    }
    if (m == 0) { output[p] = 0.0f; output_masks[p] = 0; return; }
    if (m == 1) { output[p] = first; output_masks[p] = 1; return; }
    if (m == 2) {
        const float wsum = first_w + second_w;
        output[p] = wsum > 0.0f ? (first_w * first + second_w * second) / wsum
                                : 0.5f * (first + second);
        output_masks[p] = 1;
        return;
    }

    float mu = stack_median(scratch, p, np, m);
    for (unsigned int i = 0; i < m; ++i) {
        const unsigned int idx = i * np + p;
        scratch[idx] = fabsf(scratch[idx] - mu);
    }
    const float mad = stack_median(scratch, p, np, m);
    float sigma = stack_max(1.4826f * mad, 1e-4f * stack_max(fabsf(mu), 1.0f));

    // Always reload the original frame-major values for clipping and inlier sums.
    for (unsigned int iter = 0; iter < 3; ++iter) {
        const float low = mu - 3.0f * sigma;
        const float high = mu + 3.0f * sigma;
        float wsum = 0.0f, wysum = 0.0f;
        for (unsigned int k = 0; k < nf; ++k) {
            const unsigned int idx = k * np + p;
            if (masks[idx]) {
                const float w = weights[k];
                wsum += w;
                wysum += w * stack_clamp(values[idx], low, high);
            }
        }
        if (wsum <= 0.0f) break;
        const float next_mu = wysum / wsum;
        float variance = 0.0f;
        for (unsigned int k = 0; k < nf; ++k) {
            const unsigned int idx = k * np + p;
            if (masks[idx]) {
                const float diff = stack_clamp(values[idx], low, high) - next_mu;
                variance += weights[k] * diff * diff;
            }
        }
        mu = next_mu;
        sigma = stack_max(1.134f * sqrtf(stack_max(variance / wsum, 0.0f)),
                          1e-4f * stack_max(fabsf(mu), 1.0f));
    }
    const float low = mu - 3.0f * sigma;
    const float high = mu + 3.0f * sigma;
    float wsum = 0.0f, vsum = 0.0f;
    for (unsigned int k = 0; k < nf; ++k) {
        const unsigned int idx = k * np + p;
        if (masks[idx]) {
            const float v = values[idx];
            if (v >= low && v <= high) {
                const float w = weights[k];
                wsum += w;
                vsum += w * v;
            }
        }
    }
    output[p] = wsum > 0.0f ? vsum / wsum : mu;
    output_masks[p] = 1;
}
