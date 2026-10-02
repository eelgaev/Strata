// include/strata/core/weight_form.hpp - the kernels' WForm of a BF16-form weight as the table holds it: the pack's
// BF16, or under STRATA_FP16=load FP16 (`F16InF32`) or F32 (the gates).
#pragma once

#include "strata/core/weights.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include <stdexcept>
#include <string>

namespace strata::core {

inline strata::kernels::WForm wform(const WeightRef* w, const char* what) {
    switch (w->kind) {
    case WeightKind::Bf16InF32: return strata::kernels::WForm::Bf16;
    case WeightKind::F16InF32: return strata::kernels::WForm::F16;
    case WeightKind::F32: return strata::kernels::WForm::F32;
    default:
        throw std::runtime_error(std::string(what) + " is engine form " + std::to_string((int) w->kind) +
                                 ", not BF16, FP16 or F32");
    }
}

}  // namespace strata::core
