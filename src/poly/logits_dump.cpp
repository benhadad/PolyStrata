// src/poly/logits_dump.cpp - the engine's logits for a list of token ids, as raw floats, to compare with
// llama.cpp's (any architecture the engine runs).  Run by hand; see usage() for the options.
//
// The prompt goes in calls of max_rows() tokens (eval(), the last one with logits); then each --decode token goes
// in alone.  DIR/prompt_logits.f32 gets the prompt's last logits, DIR/decode_logits.f32 one row a decode token
// (n_vocab() floats a row, or --vocab N's first N).  The five likeliest of each row go to stdout.
#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* const kUsage =
    "usage: poly_logits_dump MODEL.gguf (--tokens ID,ID,... | --tokens-file FILE) [options]\n"
    "  --tokens ID,ID,...     the prompt's token ids\n"
    "  --tokens-file FILE     the prompt's ids from a file, apart by commas, white space or brackets (a JSON list)\n"
    "  --decode ID,ID,...     tokens fed one at a time after the prompt, each with its logits\n"
    "  --out DIR              where the .f32 files go (default .)\n"
    "  --vocab N              write a row's first N scores (llama.cpp's unpadded vocabulary)\n"
    "  --max-context N        the engine's context (default 4096)\n"
    "  --prompt-batch N       the prompt's tokens a call\n"
    "  --kv fp16|q8_0         the attention cache's format (default fp16)\n"
    "  --expert-profile FILE  the expert cache's profile, as strata-poly's\n"
    "  --threads N            the CPU experts' threads\n"
    "  --no-graphs            no recorded launches\n"
    "  --no-gpu-experts       every routed expert on the CPU\n"
    "  --reload               after the prompt, save the state and rows, reset, feed other tokens, load them back\n";

std::vector<int32_t> ids(const std::string& s) {
    std::vector<int32_t> out;
    for (const char* p = s.c_str(); *p;) {
        char* end = nullptr;
        out.push_back((int32_t) std::strtol(p, &end, 10));
        if (end == p) throw std::runtime_error("not a list of token ids: " + s);
        p = *end == ',' ? end + 1 : end;
    }
    return out;
}

// ids apart by commas, white space or brackets
std::vector<int32_t> ids_in_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string t = ss.str(), u;
    for (char& c : t)
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '[' || c == ']') c = ',';
    for (char c : t)
        if (!(c == ',' && (u.empty() || u.back() == ','))) u += c;
    while (!u.empty() && u.back() == ',') u.pop_back();
    return ids(u);
}

void top(const std::string& what, const float* l, int n) {
    std::vector<int> o((size_t) n);
    std::iota(o.begin(), o.end(), 0);
    const int k = std::min(5, n);
    std::partial_sort(o.begin(), o.begin() + k, o.end(), [l](int a, int b) { return l[a] > l[b]; });
    std::printf("%s:", what.c_str());
    for (int i = 0; i < k; ++i) std::printf(" %d(%.4f)", o[(size_t) i], l[o[(size_t) i]]);
    std::printf("\n");
}

FILE* create(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error(kUsage);
        strata::poly::Options o;
        o.model = argv[1];
        o.max_context = 4096;
        o.window = 1;
        std::vector<int32_t> prompt, decode;
        std::string out = ".";
        bool reload = false;
        int vocab = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> const char* {
                if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
                return argv[++i];
            };
            if (a == "--tokens") prompt = ids(val());
            else if (a == "--tokens-file") prompt = ids_in_file(val());
            else if (a == "--decode") decode = ids(val());
            else if (a == "--out") out = val();
            else if (a == "--vocab") vocab = std::atoi(val());
            else if (a == "--max-context") o.max_context = std::atoi(val());
            else if (a == "--prompt-batch") o.prompt_batch = std::atoi(val());
            else if (a == "--kv") {
                const std::string f = val();
                if (f == "q8_0") o.kv_format = strata::poly::KvFmt::kQ8_0;
                else if (f != "fp16") throw std::runtime_error("--kv is fp16 or q8_0");
            } else if (a == "--expert-profile") o.profile = val();
            else if (a == "--threads") o.threads = std::atoi(val());
            else if (a == "--no-graphs") o.graphs = false;
            else if (a == "--no-gpu-experts") o.gpu_experts = false;
            else if (a == "--reload") reload = true;
            else throw std::runtime_error("unknown option " + a + "\n" + kUsage);
        }
        if (prompt.empty()) throw std::runtime_error(std::string("no prompt tokens\n") + kUsage);

        auto e = strata::poly::Engine::open(o);
        std::fprintf(stderr, "loaded: %lld MiB VRAM free, %d of %d experts cached (%.2f GiB)\n",
                     (long long) e->vram_free_mib(), e->cached_experts(), e->total_experts(), e->cached_gib());
        const int nv = e->n_vocab(), nw = vocab > 0 ? std::min(vocab, nv) : nv;
        for (size_t i = 0; i < prompt.size(); i += (size_t) e->max_rows()) {
            const int m = (int) std::min(prompt.size() - i, (size_t) e->max_rows());
            e->eval(prompt.data() + i, m, i + (size_t) m == prompt.size());
        }
        FILE* f = create(out + "/prompt_logits.f32");
        std::fwrite(e->logits(), sizeof(float), (size_t) nw, f);
        std::fclose(f);
        top("prompt", e->logits(), nv);
        if (reload) {   // what a conversation checkpoint does (save_state, save_rows), and back
            const int64_t pos = e->pos();
            std::vector<uint8_t> state(e->state_bytes()), rows((size_t) pos * e->row_bytes());
            e->save_state(state.data());
            e->save_rows(0, pos, rows.data());
            e->reset();
            const int32_t other[4] = {11, 12, 13, 14};   // something else in the caches and state meanwhile
            e->eval(other, 4, false);
            e->load_rows(rows.data(), 0, pos, 0, pos);
            e->load_state(state.data(), pos);
            std::fprintf(stderr, "reload: %zu bytes of state, %zu of rows for %lld positions\n", state.size(),
                         rows.size(), (long long) pos);
        }
        f = create(out + "/decode_logits.f32");
        for (size_t k = 0; k < decode.size(); ++k) {
            e->eval(&decode[k], 1, true);
            std::fwrite(e->logits(), sizeof(float), (size_t) nw, f);
            top("decode " + std::to_string(k), e->logits(), nv);
        }
        std::fclose(f);
        const strata::poly::Stats& st = e->stats();
        std::fprintf(stderr, "experts from VRAM %lld (%.1f%%), CPU %lld; %d of %d experts cached (%.2f GiB)\n",
                     (long long) st.gpu_experts,
                     100.0 * (double) st.gpu_experts / (double) std::max<int64_t>(1, st.gpu_experts + st.cpu_experts),
                     (long long) st.cpu_experts, e->cached_experts(), e->total_experts(), e->cached_gib());
        return 0;
    } catch (const std::exception& x) {
        std::fprintf(stderr, "poly_logits_dump: %s\n", x.what());
        return 1;
    }
}
