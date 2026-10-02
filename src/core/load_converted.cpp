// src/core/load_converted.cpp - see include/strata/core/load_converted.hpp.
#include "strata/core/load_converted.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <thread>

namespace strata::core {

bool load_converted_kind(const std::string& name, bool& gate) {
    static const char* f16[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_ffn_down.weight",
                                ".hc_ffn_up.weight", ".ple_value.weight", ".indexer.k_proj.weight",
                                ".indexer.q_proj.weight"};
    static const char* gates[] = {".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight", ".hc_attn_inject.weight",
                                  ".hc_ffn_inject.weight", ".ssm_alpha.weight", ".ssm_beta.weight"};
    gate = false;
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;
    if (name.rfind("blk.", 0) != 0) return false;
    for (const char* s : f16) if (name.ends_with(s)) return true;
    for (const char* s : gates) if (name.ends_with(s)) { gate = true; return true; }
    return false;
}

namespace {
constexpr uint32_t kF32 = 0, kF16 = 1, kQ8_0 = 8, kBF16 = 30;
}

bool LoadConverted::build(const std::vector<std::string>& shards, bool gates_fp16, std::string& err) {
    host_.clear();
    tensors_f16 = tensors_f32 = 0;
    bytes = subnormal = flushed = 0;
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            struct Job { const strata::TensorInfo* t; size_t h; };
            std::vector<Job> jobs;
            for (const auto& t : gguf.tensors()) {
                bool gate = false;
                if (!load_converted_kind(t.name, gate)) continue;
                if (t.type != kQ8_0 && t.type != kBF16 && t.type != kF16 && t.type != kF32) {
                    err = "STRATA_FP16=load: " + t.name + " is " + t.type_name() + " in the GGUF (Q8_0, BF16, F16 "
                          "or F32 expected)";
                    return false;
                }
                if (t.shape.empty() || t.shape.size() > 2 || (t.type == kQ8_0 && t.shape[0] % 32 != 0)) {
                    err = "STRATA_FP16=load: " + t.name + " has an unexpected shape";
                    return false;
                }
                Host h;
                h.name = t.name;
                h.ne0 = (int64_t) t.shape[0];
                h.ne1 = t.shape.size() > 1 ? (int64_t) t.shape[1] : 0;
                h.f32 = gate && !gates_fp16 && t.type == kF32;
                h.data.resize(t.elements() * (h.f32 ? 4 : 2));
                jobs.push_back(Job{&t, host_.size()});
                host_.push_back(std::move(h));
            }
            // the conversions, in parallel over tensors (the GGUF stays mapped for this loop)
            std::atomic<size_t> next{0};
            std::atomic<uint64_t> sub{0}, fl{0}, over{0};
            auto work = [&] {
                for (size_t j; (j = next.fetch_add(1)) < jobs.size();) {
                    const strata::TensorInfo& t = *jobs[j].t;
                    Host& h = host_[jobs[j].h];
                    const uint8_t* src = gguf.tensor_data(t);
                    const uint64_t n = t.elements();
                    if (h.f32) { std::memcpy(h.data.data(), src, n * 4); continue; }
                    uint16_t* dst = reinterpret_cast<uint16_t*>(h.data.data());
                    uint64_t s = 0, f = 0, o = 0;
                    auto put = [&](uint64_t i, float x) {
                        if (std::fabs(x) > 65504.0f) ++o;
                        const uint16_t v = strata::kernels::f16_from_f32(x);
                        if (x != 0.0f) {
                            if ((v & 0x7fffu) == 0) ++f;
                            else if ((v & 0x7c00u) == 0) ++s;
                        }
                        dst[i] = v;
                    };
                    if (t.type == kQ8_0) {
                        for (uint64_t b = 0; b < n / 32; ++b) {
                            const uint8_t* blk = src + b * 34;
                            uint16_t dh;
                            std::memcpy(&dh, blk, 2);
                            const float d = strata::kernels::f32_from_f16(dh);
                            for (int k = 0; k < 32; ++k) put(b * 32 + k, d * (float) (int8_t) blk[2 + k]);
                        }
                    } else if (t.type == kBF16) {
                        for (uint64_t i = 0; i < n; ++i) {
                            uint16_t v;
                            std::memcpy(&v, src + 2 * i, 2);
                            put(i, strata::kernels::f32_from_bf16(v));
                        }
                    } else if (t.type == kF16) {
                        std::memcpy(dst, src, n * 2);
                    } else {
                        for (uint64_t i = 0; i < n; ++i) {
                            float x;
                            std::memcpy(&x, src + 4 * i, 4);
                            put(i, x);
                        }
                    }
                    sub += s; fl += f; over += o;
                }
            };
            std::vector<std::thread> pool;
            const unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
            for (unsigned i = 0; i < nt; ++i) pool.emplace_back(work);
            for (auto& th : pool) th.join();
            if (over) {
                err = "STRATA_FP16=load: " + std::to_string((unsigned long long) over.load()) +
                      " weight values are beyond FP16's range (65504) in " + path;
                return false;
            }
            subnormal += sub;
            flushed += fl;
        }
    } catch (const std::exception& e) {
        err = std::string("STRATA_FP16=load: ") + e.what();
        return false;
    }
    if (host_.empty()) { err = "STRATA_FP16=load: none of the BF16-form weights are in the GGUF shards"; return false; }
    for (const auto& h : host_) {
        (h.f32 ? tensors_f32 : tensors_f16)++;
        bytes += h.data.size();
    }
    return true;
}

void LoadConverted::served_names(std::set<std::string>& out) const {
    for (const auto& h : host_) out.insert(h.name);
}

bool LoadConverted::attach(WeightTable& table, std::vector<void*>& owner, std::string& err) const {
    for (const auto& h : host_) {
        auto it = table.table_.find(h.name);
        if (it == table.table_.end()) { err = "STRATA_FP16=load: " + h.name + " is not in the pack"; return false; }
        WeightRef& r = it->second;
        if (r.kind != WeightKind::Bf16InF32 || r.ne0 != h.ne0 || r.ne1 != h.ne1) {
            err = "STRATA_FP16=load: " + h.name + " is not the pack's BF16 row of the same shape (form " +
                  std::to_string((int) r.kind) + ", " + std::to_string((long long) r.ne0) + " x " +
                  std::to_string((long long) r.ne1) + ")";
            return false;
        }
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, h.data.size());
        if (e == cudaSuccess) {
            owner.push_back(p);
            e = cudaMemcpy(p, h.data.data(), h.data.size(), cudaMemcpyHostToDevice);
        }
        if (e != cudaSuccess) { err = "STRATA_FP16=load: upload of " + h.name + ": " + cudaGetErrorString(e); return false; }
        r.data = p;
        r.bytes = h.data.size();
        r.kind = h.f32 ? WeightKind::F32 : WeightKind::F16InF32;
        r.resident = true;
    }
    return true;
}

void LoadConverted::free_all(std::vector<void*>& owner) {
    for (void* p : owner) cudaFree(p);
    owner.clear();
}

}  // namespace strata::core
