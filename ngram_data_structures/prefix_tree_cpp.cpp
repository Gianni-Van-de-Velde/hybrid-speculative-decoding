#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

struct PrefixTreeNode {
    std::unordered_map<int64_t, std::unique_ptr<PrefixTreeNode>> children;
    std::vector<size_t> starts; // positions (i+1) where a continuation begins
};

class PrefixTree {
public:
    PrefixTree(size_t max_match_length = 10, size_t max_continuation_length = 10)
        : max_match_length_(max_match_length), max_continuation_length_(max_continuation_length) {
        root_ = std::make_unique<PrefixTreeNode>();
    }

    // new_context: iterable of Python objects (tokens). We store original py::object
    void insert_incremental(py::iterable new_context) {
        size_t start_idx = context_vals_.size();
        for (auto item : new_context) {
            int64_t v = item.cast<int64_t>();
            context_vals_.push_back(v);
        }

        for (size_t i = start_idx; i < context_vals_.size(); ++i) {
            PrefixTreeNode* node = root_.get();
            size_t start_position = 0;
            if (i + 1 > max_match_length_) {
                start_position = i + 1 - max_match_length_;
            }
            // iterate reversed from context[start_position..i]
            for (size_t t = i + 1; t-- > start_position; ) {
                int64_t key = context_vals_[t];
                auto it = node->children.find(key);
                if (it == node->children.end()) {
                    node->children[key] = std::make_unique<PrefixTreeNode>();
                    it = node->children.find(key);
                }
                node = it->second.get();
                node->starts.push_back(i + 1); // store position after match
            }
        }
    }

    // tokens: iterable of Python objects to search for matches against suffixes of context
    std::pair<std::vector<int64_t>, std::vector<int>> top_k_unique_matches(py::iterable tokens) {
        std::vector<int64_t> token_vals;
        for (auto it : tokens) token_vals.push_back(it.cast<int64_t>());

        PrefixTreeNode* node = root_.get();
        std::unordered_map<size_t, size_t> best_matches; // pos -> length

        // search reversed to align rightmost token; consider only last max_match_length tokens
        size_t consider = std::min(token_vals.size(), max_match_length_);
        for (size_t idx = 0; idx < consider; ++idx) {
            int64_t t = token_vals[token_vals.size() - 1 - idx];
            auto it = node->children.find(t);
            if (it != node->children.end()) {
                node = it->second.get();
                for (const auto &pos : node->starts) {
                    size_t len = idx + 1;
                    auto existing = best_matches.find(pos);
                    if (existing == best_matches.end() || len > existing->second) {
                        best_matches[pos] = len;
                    }
                }
            } else {
                break;
            }
        }

        // Build spec_tokens and parents using original context objects
        std::vector<int64_t> spec_tokens;
        std::vector<int> parents;

        std::vector<size_t> positions;
        positions.reserve(best_matches.size());
        for (auto &kv : best_matches) positions.push_back(kv.first);

        for (size_t pos : positions) {
            size_t start_idx = pos; // pos is i+1 in the Python code
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
    std::unique_ptr<PrefixTreeNode> root_;
    size_t max_match_length_;
    size_t max_continuation_length_;
    std::vector<int64_t> context_vals_;
};

PYBIND11_MODULE(prefix_tree_cpp, m) {
    m.doc() = "PrefixTree implemented in C++ with pybind11";

    py::class_<PrefixTree>(m, "PrefixTree")
        .def(py::init<size_t, size_t>(), py::arg("max_match_length") = 10, py::arg("max_continuation_length") = 10)
        .def("insert_incremental", &PrefixTree::insert_incremental, py::arg("new_context"))
        .def("top_k_unique_matches", &PrefixTree::top_k_unique_matches, py::arg("tokens"));
}
