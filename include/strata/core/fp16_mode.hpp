// include/strata/core/fp16_mode.hpp - STRATA_FP16: where the model's BF16-form weights run as FP16 (V100: no BF16
// tensor cores; cuBLAS BF16 GEMMs are 3-18x slower than FP16 there).
//
//   unset / off  the BF16 forms as the pack holds them (the default: byte-identical outputs)
//   prefill      the prompt path converts each BF16 weight to FP16 right before its GEMM; decode is untouched
//   load         at load, before serving, the weights are taken from the GGUF instead of the pack's BF16 rows:
//                Q8_0 / BF16 / F16 sources become FP16 and the gates (router, shared gate, hc inject, SSM
//                alpha/beta: F32 in the GGUF) stay F32; every reader uses those forms.  STRATA_FP16_GATES=1
//                makes the gates FP16 too.  The MTP drafter keeps its BF16 weights.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata {

enum class Fp16Mode { Off, Prefill, Load };

inline Fp16Mode fp16_mode() {
    static const Fp16Mode m = [] {
        const char* e = std::getenv("STRATA_FP16");
        if (e == nullptr || *e == 0 || std::strcmp(e, "off") == 0 || std::strcmp(e, "0") == 0) return Fp16Mode::Off;
        if (std::strcmp(e, "prefill") == 0) return Fp16Mode::Prefill;
        if (std::strcmp(e, "load") == 0) return Fp16Mode::Load;
        std::fprintf(stderr, "STRATA_FP16=%s: expected off, prefill or load\n", e);
        std::exit(2);
    }();
    return m;
}

/// STRATA_FP16_GATES=1 (with STRATA_FP16=load): the F32 gates become FP16 at load too.
inline bool fp16_gates() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_FP16_GATES");
        return e != nullptr && e[0] == '1';
    }();
    return v && fp16_mode() == Fp16Mode::Load;
}

/// STRATA_FP16_MTP=1: the MTP drafter's prompt-path GEMMs (its attention hyper-connection down/up, run per prompt chunk)
/// as FP16: those two BF16 weights converted to FP16 when the drafter loads, FP16 activations.  Its decode is unchanged
/// (FP32 activations).  Independent of STRATA_FP16.
inline bool fp16_mtp() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_FP16_MTP");
        return e != nullptr && e[0] == '1';
    }();
    return v;
}

}  // namespace strata
