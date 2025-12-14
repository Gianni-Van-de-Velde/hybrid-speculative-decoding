#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <set>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <limits>

namespace py = pybind11;

// We'll use a 64-bit polynomial rolling hash (mod 2^64, wrap-around) for
// O(1) substring hashing using prefix/power arrays. This is fast and simple
// for short n-grams.
static const uint64_t HASH_BASE = 11400714819323198485ULL; // odd large constant
static inline uint64_t mix_token(uint64_t v) {
    // Mix token value a bit to avoid small-value collisions; simple xor+mul
    v += 0x9e3779b97f4a7c15ULL;
    v ^= (v >> 23);
    v *= 0xC2B2AE3D27D4EB4FULL;
    return v;
}

class HashMapNgram {
public:
    HashMapNgram(size_t max_match_length = 10, size_t continuation_length = 10)
        : max_match_length_(max_match_length), continuation_length_(continuation_length) {
        // index 0 unused; use indices 1..max_match_length_
        copy_dict_.resize(max_match_length_ + 1);
        // prepare power table for rolling hashes
        pow_.resize(max_match_length_ + 1);
        pow_[0] = 1ULL;
        for (size_t i = 1; i <= max_match_length_; ++i) pow_[i] = pow_[i - 1] * HASH_BASE;
        // prefix_hash_ starts with 0
        prefix_hash_.clear();
        prefix_hash_.push_back(0ULL);
    }

    void preprocess_prompt(py::iterable input_ids) {
        size_t prev_len = context_.size();
        // append new tokens
        for (auto it : input_ids) {
            int64_t v = it.cast<int64_t>();
            context_.push_back(v);
            // update prefix hash incrementally
            uint64_t mixed = mix_token(static_cast<uint64_t>(v));
            uint64_t new_h = prefix_hash_.back() * HASH_BASE + mixed;
            prefix_hash_.push_back(new_h);
        }
        // update hash tables for each k using O(1) substring hash via prefix hashes
        for (size_t k = 1; k <= max_match_length_; ++k) {
            if (context_.size() < k) continue;

            size_t start_i = 0;
            if (prev_len > 0 && prev_len > k) start_i = prev_len - k;
            size_t max_i = context_.size() - k;
            auto &map_k = copy_dict_[k];
            // Heuristic reserve: expect roughly (context_.size()/k) distinct n-grams
            if (map_k.empty()) map_k.reserve(std::max<size_t>(16, context_.size() / (k + 1)));

            for (size_t i = start_i; i <= max_i; ++i) {
                // substring hash: h = prefix_hash_[i+k] - prefix_hash_[i] * pow_[k]
                uint64_t h = prefix_hash_[i + k] - prefix_hash_[i] * pow_[k];
                map_k[h].push_back(static_cast<uint32_t>(i + k)); // store position i+k
            }
        }
    }

    std::pair<std::vector<int64_t>, std::vector<int>> find_text_position(py::iterable input_ids) {
        std::vector<int64_t> token_vals;
        for (auto it : input_ids) token_vals.push_back(it.cast<int64_t>());

        // best_matches: pos -> length (choose longest match for each pos)
        std::unordered_map<uint32_t, size_t> best_matches;

        for (size_t k = 1; k <= max_match_length_; ++k) {
            if (token_vals.size() < k) continue;
            // compute hash of last k tokens
            size_t off = token_vals.size() - k;
            uint64_t h = 0ULL;
            for (size_t j = off; j < token_vals.size(); ++j) {
                h = h * HASH_BASE + mix_token(static_cast<uint64_t>(token_vals[j]));
            }

            auto &map_k = copy_dict_[k];
            auto it = map_k.find(h);
            if (it != map_k.end()) {
                for (auto pos : it->second) {
                    auto existing = best_matches.find(pos);
                    if (existing == best_matches.end() || k > existing->second) {
                        best_matches[pos] = k;
                    }
                }
            }
        }

        // Build spec_tokens and parents using original context objects
        std::vector<int64_t> spec_tokens;
        std::vector<int> parents;

        std::vector<uint32_t> positions;
        positions.reserve(best_matches.size());
        for (auto &kv : best_matches) positions.push_back(kv.first);

        for (uint32_t pos : positions) {
            size_t start_idx = static_cast<size_t>(pos); // pos is i+1 in the Python code
            for (size_t j = 0; j < continuation_length_; ++j) {
                if (start_idx + j >= context_.size()) break;
                spec_tokens.push_back(context_[start_idx + j]);
                if (j == 0) {
                    parents.push_back(-1);
                } else {
                    parents.push_back(static_cast<int>(spec_tokens.size()) - 2);
                }
            }
        }

        return {spec_tokens, parents};
    }

private:
    size_t max_match_length_;
    size_t continuation_length_;
    std::vector<int64_t> context_;
    // index by k: copy_dict_[k] -> map from hash -> vector<position>
    std::vector<std::unordered_map<uint64_t, std::vector<uint32_t>>> copy_dict_;
    // rolling/hash helpers
    std::vector<uint64_t> prefix_hash_;
    std::vector<uint64_t> pow_;
};

PYBIND11_MODULE(hashmap_cpp, m) {
    m.doc() = "HashMap-based ngram copy detector implemented in C++ with pybind11";

    py::class_<HashMapNgram>(m, "HashMapNgram")
        .def(py::init<size_t, size_t>(), py::arg("max_match_length") = 10, py::arg("continuation_length") = 10)
        .def("preprocess_prompt", &HashMapNgram::preprocess_prompt, py::arg("input_ids"))
        .def("find_text_position", &HashMapNgram::find_text_position, py::arg("input_ids"));
}
