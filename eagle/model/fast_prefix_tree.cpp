#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace py = pybind11;

class PrefixTreeNode {
public:
    std::unordered_map<int, PrefixTreeNode*> children;
    std::vector<int> starts;

    ~PrefixTreeNode() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
};

class PrefixTree {
public:
    PrefixTree(int max_match_length = 10)
        : max_match_length(max_match_length) {
        root = new PrefixTreeNode();
    }

    ~PrefixTree() {
        delete root;
    }

    void insert_incremental(const std::vector<int>& new_context) {
        int start_idx = context.size();
        context.insert(context.end(), new_context.begin(), new_context.end());

        for (int i = start_idx; i < (int)context.size(); ++i) {
            PrefixTreeNode* node = root;
            int start_position = std::max(i - max_match_length + 1, 0);
            for (int j = i; j >= start_position; --j) {
                int t = context[j];
                if (node->children.find(t) == node->children.end()) {
                    node->children[t] = new PrefixTreeNode();
                }
                node = node->children[t];
                node->starts.push_back(i + 1);
            }
        }
    }

    std::vector<std::pair<int, int>> top_k_unique_matches(const std::vector<int>& tokens, int nr_continuations = 3, int minimal_match_length = 1) {
        PrefixTreeNode* node = root;
        std::unordered_map<int, int> best_matches; // pos -> length

        int len = (int)tokens.size();
        int start = std::max(len - max_match_length, 0);
        for (int i = 0; i < len - start; ++i) {
            int t = tokens[len - 1 - i];
            if (node->children.find(t) != node->children.end()) {
                node = node->children[t];
                for (int pos : node->starts) {
                    if (best_matches.find(pos) == best_matches.end() || i + 1 > best_matches[pos]) {
                        best_matches[pos] = i + 1;
                    }
                }
            } else {
                break;
            }
        }

        std::vector<std::pair<int, int>> sorted_matches(best_matches.begin(), best_matches.end());
        std::sort(sorted_matches.begin(), sorted_matches.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second;
        });

        std::vector<std::pair<int, int>> filtered_matches;
        for (const auto& p : sorted_matches) {
            if (p.second >= minimal_match_length) {
                filtered_matches.push_back(p);
            }
        }

        if ((int)filtered_matches.size() > nr_continuations) {
            filtered_matches.resize(nr_continuations);
        }

        return filtered_matches;
    }

public:
    std::vector<int> context;

private:
    PrefixTreeNode* root;
    int max_match_length;
};

PYBIND11_MODULE(fast_prefix_tree, m) {
    py::class_<PrefixTree>(m, "PrefixTree")
        .def(py::init<int>(), py::arg("max_match_length") = 10)
        .def("insert_incremental", &PrefixTree::insert_incremental)
        .def("top_k_unique_matches", &PrefixTree::top_k_unique_matches,
             py::arg("tokens"), py::arg("nr_continuations") = 3, py::arg("minimal_match_length") = 1);
}
