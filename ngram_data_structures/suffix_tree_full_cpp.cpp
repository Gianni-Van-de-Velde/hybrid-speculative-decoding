#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

namespace py = pybind11;

struct SuffixTreeNode {
    std::unordered_map<int64_t, std::unique_ptr<SuffixTreeNode>> children;
    std::vector<size_t> ends;
};

class SuffixTreeFull {
public:
    SuffixTreeFull(size_t max_match_length = 10, size_t max_continuation_length = 10)
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
            if (i + 1 > max_match_length_ + max_continuation_length_) {
                start_position = i + 1 - (max_match_length_ + max_continuation_length_);
            }
            // insert suffix up to window: iterate forward from start_position..i
            for (size_t j = start_position; j <= i; ++j) {
                int64_t key = context_vals_[j];
                auto it = node->children.find(key);
                if (it == node->children.end()) {
                    node->children[key] = std::make_unique<SuffixTreeNode>();
                    it = node->children.find(key);
                }
                node = it->second.get();
                node->ends.push_back(i + 1);
            }
        }
    }

    std::pair<std::vector<int64_t>, std::vector<int>> top_k_unique_matches(py::iterable tokens) {
        std::vector<int64_t> token_vals;
        for (auto it : tokens) token_vals.push_back(it.cast<int64_t>());

        std::unordered_set<size_t> seen_ends;
        std::vector<int64_t> spec_tokens;
        std::vector<int> parents;

        for (size_t len = max_match_length_; len > 0; --len) {
            if (token_vals.size() < len) continue;
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
            if (!matched) continue;

            // if seen_ends is a superset of node->ends, skip
            bool superset = true;
            for (const auto &e : node->ends) {
                if (seen_ends.find(e) == seen_ends.end()) { superset = false; break; }
            }
            if (superset) continue;

            speculate_tree(node, spec_tokens, parents, -1, seen_ends, 0);
            // update seen_ends with node->ends
            for (const auto &e : node->ends) seen_ends.insert(e);
        }

        return {spec_tokens, parents};
    }

private:
    void speculate_tree(SuffixTreeNode* node, std::vector<int64_t> &tokens_out, std::vector<int> &parents_out, int parent, const std::unordered_set<size_t> &seen_ends, size_t depth) {
        if (depth >= max_continuation_length_) return;

        for (auto &kv : node->children) {
            int64_t child_token = kv.first;
            SuffixTreeNode* child_node = kv.second.get();
            bool skip = false;
            for (const auto &end : child_node->ends) {
                if (seen_ends.find(end - 1) != seen_ends.end()) { skip = true; break; }
            }
            if (skip) continue;
            tokens_out.push_back(child_token);
            parents_out.push_back(parent);
            int new_parent = static_cast<int>(tokens_out.size()) - 1;
            speculate_tree(child_node, tokens_out, parents_out, new_parent, seen_ends, depth + 1);
        }
    }

    std::unique_ptr<SuffixTreeNode> root_;
    size_t max_match_length_;
    size_t max_continuation_length_;
    std::vector<int64_t> context_vals_;
};

PYBIND11_MODULE(suffix_tree_full_cpp, m) {
    m.doc() = "SuffixTree full implemented in C++ with pybind11";

    py::class_<SuffixTreeFull>(m, "SuffixTree")
        .def(py::init<size_t, size_t>(), py::arg("max_match_length") = 10, py::arg("max_continuation_length") = 10)
        .def("insert_incremental", &SuffixTreeFull::insert_incremental, py::arg("new_context"))
        .def("top_k_unique_matches", &SuffixTreeFull::top_k_unique_matches, py::arg("tokens"));
}
