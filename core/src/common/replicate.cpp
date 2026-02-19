#include <common/replicate.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace ss {
    std::vector<IngestConfig> expand_replicas(const std::vector<IngestConfig>& in) {
        std::vector<IngestConfig> out;
        out.reserve(in.size());

        for (const auto& s : in) {
            const int n = std::max(1, s.replicate.count);

            if (n == 1) {
                auto one = s;
                one.replicate.count = 1;
                one.replicate.ids.clear();
                out.push_back(std::move(one));
                continue;
            }

            std::vector<std::string> ids = s.replicate.ids;
            if (ids.size() < static_cast<size_t>(n)) {
                ids.reserve(static_cast<size_t>(n));
                for (size_t i = ids.size(); i < static_cast<size_t>(n); ++i) {
                    ids.push_back(s.id + "_" + std::to_string(i));
                }
            }

            for (int i = 0; i < n; ++i) {
                auto r = s;
                r.id = ids[static_cast<size_t>(i)];
                r.replicate.count = 1;
                r.replicate.ids.clear();
                out.push_back(std::move(r));
            }
        }

        return out;
    }
}
