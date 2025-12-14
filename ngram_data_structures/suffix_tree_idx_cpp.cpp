#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

struct SuffixTreeNode {
    std::unordered_map<int64_t, std::unique_ptr<SuffixTreeNode>> children;
    std::vector<size_t> starts; // positions (index+1)
};

class SuffixTreeIdx {
public:
    SuffixTreeIdx(size_t max_match_length = 10, size_t max_continuation_length = 10)
        : max_match_length_(max_match_length), max_continuation_length_(max_continuation_length) {
        root_ = std::make_unique<SuffixTreeNode>();
    }

    void insert_incremental(py::iterable new_context) {
        size_t start_idx = context_vals_.size();
        for (auto item : new_context) {
            int64_t v = item.cast<int64_t>();
            context_vals_.push_back(v);
        }

        for (size_t i = start_idx; i < context_vals_.size(); ++i) {
            SuffixTreeNode* node = root_.get();
            size_t start_position = 0;
            if (i + 1 > max_match_length_) {
                start_position = i + 1 - max_match_length_;
            }
            // insert suffix up to max_match_length: iterate forward from start_position..i
            for (size_t j = start_position; j <= i; ++j) {
                int64_t key = context_vals_[j];
                auto it = node->children.find(key);
                if (it == node->children.end()) {
                    node->children[key] = std::make_unique<SuffixTreeNode>();
                    it = node->children.find(key);
                }
                node = it->second.get();
                // +1 to align with Python behavior: store position after the matched token
                node->starts.push_back(j + 1);
            }
        }
    }

    std::pair<std::vector<int64_t>, std::vector<int>> top_k_unique_matches(py::iterable tokens) {
        std::vector<int64_t> token_vals;
        for (auto it : tokens) token_vals.push_back(it.cast<int64_t>());

        std::unordered_map<size_t, size_t> best_matches; // pos -> length

        // for lengths 1..max_match_length, check whether tokens[-i:] match
        for (size_t len = 1; len <= max_match_length_; ++len) {
            if (token_vals.size() < len) break;
            SuffixTreeNode* node = root_.get();
            size_t start = token_vals.size() - len;
            bool matched = true;
            for (size_t k = start; k < token_vals.size(); ++k) {
                int64_t t = token_vals[k];
                auto it = node->children.find(t);
                if (it != node->children.end()) {
                    node = it->second.get();
                } else {
                    matched = false;
                    break;
                }
            }
            if (matched) {
                for (const auto &pos : node->starts) {
                    best_matches[pos] = len; // overwrite with current len (increasing len overwrites shorter)
                }
            }
        }

        // Build spec_tokens and parents using original context values
        std::vector<int64_t> spec_tokens;
        std::vector<int> parents;

        std::vector<size_t> positions;
        positions.reserve(best_matches.size());
        for (auto &kv : best_matches) positions.push_back(kv.first);

        for (size_t pos : positions) {
            size_t start_idx = pos; // pos is index+1, so start at this index
            for (size_t j = 0; j < max_continuation_length_; ++j) {
                if (start_idx + j >= context_vals_.size()) break;
                spec_tokens.push_back(context_vals_[start_idx + j]);
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
    std::unique_ptr<SuffixTreeNode> root_;
    size_t max_match_length_;
    size_t max_continuation_length_;
    std::vector<int64_t> context_vals_;
};

PYBIND11_MODULE(suffix_tree_idx_cpp, m) {
    m.doc() = "SuffixTree implemented in C++ with pybind11";

    py::class_<SuffixTreeIdx>(m, "SuffixTree")
        .def(py::init<size_t, size_t>(), py::arg("max_match_length") = 10, py::arg("max_continuation_length") = 10)
        .def("insert_incremental", &SuffixTreeIdx::insert_incremental, py::arg("new_context"))
        .def("top_k_unique_matches", &SuffixTreeIdx::top_k_unique_matches, py::arg("tokens"));
}
