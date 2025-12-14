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
    // Small-vector representation for children: pair<token, child_index>
    // child_index is -1 when not present. When branching grows, convert to map
    // mapping token -> child_index. Using indices avoids pointer indirection
    // and reduces allocation churn.
    std::vector<std::pair<int64_t, int32_t>> children_vec;
    std::unique_ptr<std::unordered_map<int64_t, int32_t>> children_map;
    // starts are stored in a global pool; this holds head index and count
    int32_t starts_head = -1; // index into global starts arrays, -1 if none
    uint32_t starts_count = 0;

    PrefixTreeNode() {
        children_vec.reserve(2);
    }

    inline int32_t find_child(int64_t key) const {
        if (children_map) {
            auto it = children_map->find(key);
            return it == children_map->end() ? -1 : it->second;
        }
        // small linear search; unrolled simple loop for speed
        const auto *data = children_vec.data();
        size_t n = children_vec.size();
        for (size_t i = 0; i < n; ++i) {
            if (data[i].first == key) return data[i].second;
        }
        return -1;
    }

    inline void insert_child(int64_t key, int32_t child_idx) {
        if (children_map) {
            (*children_map)[key] = child_idx;
            return;
        }
        // keep small vector unsorted; emplace_back is fastest for small sizes
        children_vec.emplace_back(key, child_idx);
        const size_t CONVERT_THRESHOLD = 8;
        if (children_vec.size() > CONVERT_THRESHOLD) {
            children_map = std::make_unique<std::unordered_map<int64_t, int32_t>>();
            children_map->reserve(children_vec.size() * 2);
            for (auto &p : children_vec) (*children_map)[p.first] = p.second;
            children_vec.clear();
            children_vec.shrink_to_fit();
        }
    }
};

class PrefixTreeSpeedy {
public:
    PrefixTreeSpeedy(size_t max_match_length = 10, size_t max_continuation_length = 10)
        : max_match_length_(max_match_length), max_continuation_length_(max_continuation_length) {
        // preallocate a larger pool to avoid many tiny allocations
        node_pool_.reserve(1 << 16);
        node_pool_.emplace_back();
        root_idx_ = 0;
        // reserve a modest starts pool to reduce reallocations
        starts_vals_.reserve(1 << 16);
        starts_next_.reserve(1 << 16);
    }

    // new_context: iterable of Python objects (tokens). We store raw token ids
    void insert_incremental(py::iterable new_context) {
        std::vector<int64_t> new_vals;
        new_vals.reserve(16);
        for (auto item : new_context) new_vals.push_back(item.cast<int64_t>());

        size_t start_idx = context_vals_.size();
        context_vals_.reserve(context_vals_.size() + new_vals.size());
        for (auto v : new_vals) context_vals_.push_back(v);

        for (size_t i = start_idx; i < context_vals_.size(); ++i) {
            int32_t node_idx = static_cast<int32_t>(root_idx_);
            size_t start_position = 0;
            if (i + 1 > max_match_length_) start_position = i + 1 - max_match_length_;
            for (size_t t = i + 1; t-- > start_position; ) {
                int64_t key = context_vals_[t];
                int32_t child_idx = node_pool_[node_idx].find_child(key);
                if (child_idx == -1) {
                    node_pool_.emplace_back();
                    child_idx = static_cast<int32_t>(node_pool_.size() - 1);
                    node_pool_[node_idx].insert_child(key, child_idx);
                }
                node_idx = child_idx;
                // add start to global pool: push value and link to node's head
                int32_t new_idx = static_cast<int32_t>(starts_vals_.size());
                starts_vals_.push_back(static_cast<uint32_t>(i + 1));
                starts_next_.push_back(node_pool_[node_idx].starts_head);
                node_pool_[node_idx].starts_head = new_idx;
                node_pool_[node_idx].starts_count += 1;
            }
        }
    }

    // tokens: iterable of Python objects to search for matches against suffixes of context
    std::pair<std::vector<int64_t>, std::vector<int>> top_k_unique_matches(py::iterable tokens) {
        std::vector<int64_t> token_vals;
        token_vals.reserve(16);
        for (auto it : tokens) token_vals.push_back(it.cast<int64_t>());

        int32_t node_idx = static_cast<int32_t>(root_idx_);
        std::unordered_map<size_t, size_t> best_matches; // pos -> length

        // search reversed to align rightmost token; consider only last max_match_length tokens
        size_t consider = std::min(token_vals.size(), max_match_length_);
        for (size_t idx = 0; idx < consider; ++idx) {
            int64_t t = token_vals[token_vals.size() - 1 - idx];
            int32_t child_idx = node_pool_[node_idx].find_child(t);
            if (child_idx != -1) {
                node_idx = child_idx;
                // iterate starts via global linked list
                int32_t head = node_pool_[node_idx].starts_head;
                int32_t cur = head;
                while (cur != -1) {
                    uint32_t pos = starts_vals_[cur];
                    size_t len = idx + 1;
                    auto existing = best_matches.find(pos);
                    if (existing == best_matches.end() || len > existing->second) {
                        best_matches[pos] = len;
                    }
                    cur = starts_next_[cur];
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
    // pool owns all nodes; indices in children point into this pool
    std::vector<PrefixTreeNode> node_pool_;
    size_t root_idx_ = 0;
    size_t max_match_length_;
    size_t max_continuation_length_;
    std::vector<int64_t> context_vals_;
    // global starts pool: values and next pointers (singly-linked list per node)
    std::vector<uint32_t> starts_vals_;
    std::vector<int32_t> starts_next_;
};

PYBIND11_MODULE(prefix_tree_speedy_cpp, m) {
    m.doc() = "PrefixTree implemented in C++ with pybind11";

    py::class_<PrefixTreeSpeedy>(m, "PrefixTreeSpeedy")
        .def(py::init<size_t, size_t>(), py::arg("max_match_length") = 10, py::arg("max_continuation_length") = 10)
        .def("insert_incremental", &PrefixTreeSpeedy::insert_incremental, py::arg("new_context"))
        .def("top_k_unique_matches", &PrefixTreeSpeedy::top_k_unique_matches, py::arg("tokens"));
}
 
