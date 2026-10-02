// include/strata/core/load_converted.hpp - STRATA_FP16=load: the BF16-form weights taken from the GGUF at load.
//
// The pack holds these tensors as BF16 (its `Bf16InF32` form), rounded from the GGUF's Q8_0 (hyper-connection
// down/up, output_hc, ple_value), BF16 (indexer) and F32 (router, shared gate, hc inject, SSM alpha/beta: BF16
// values in F32 containers).  Here they are read from the GGUF instead, once on the host: Q8_0 / BF16 / F16 to FP16
// (`F16InF32` form), F32 kept as F32 (`F32` form; FP16 with STRATA_FP16_GATES=1), and each GPU's table gets its own
// copy in place of the pack's row, which the arena skips.
#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;

class LoadConverted {
public:
    /// Read and convert every served tensor of `shards`.  Refuses a value FP16 cannot hold (|x| > 65504).
    bool build(const std::vector<std::string>& shards, bool gates_fp16, std::string& err);
    /// The canonical names `build` serves (for the arena's skip set).
    void served_names(std::set<std::string>& out) const;
    /// Upload into `table` (one GPU's): each row's data, bytes and kind replaced.  `owner` keeps the allocations.
    bool attach(WeightTable& table, std::vector<void*>& owner, std::string& err) const;
    static void free_all(std::vector<void*>& owner);

    size_t tensors_f16 = 0, tensors_f32 = 0;
    uint64_t bytes = 0;
    uint64_t subnormal = 0;     ///< nonzero values that became FP16 subnormals (|x| < 6.1e-5)
    uint64_t flushed = 0;       ///< nonzero values that became FP16 zero

private:
    struct Host {
        std::string name;
        int64_t ne0 = 0, ne1 = 0;
        bool f32 = false;
        std::vector<uint8_t> data;
    };
    std::vector<Host> host_;
};

/// The served kinds: true for a gate (F32 in the GGUF, kept F32 unless STRATA_FP16_GATES=1).
bool load_converted_kind(const std::string& name, bool& gate);

}  // namespace strata::core
