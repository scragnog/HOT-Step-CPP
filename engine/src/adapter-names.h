#pragma once

#include <cstring>
#include <string>

// Map a LoRA safetensors key to the GGUF base tensor name.
//
// Supported key formats (all map to GGUF "decoder.layers.0.self_attn.q_proj.weight"):
//
//   PEFT adapter_model.safetensors:
//     base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
//     base_model.model.layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (no prefix):
//     layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (diffusion_model prefix):
//     diffusion_model.layers.0.self_attn.q_proj.lora_A.weight
//
//   PEFT raw export (wrapped modules):
//     base_model.model.layers.0.mlp.original_mlp.down_proj.lora_A.default.weight
//     (the ".original_mlp." infix is stripped before matching)
//
// Steps: strip known prefix, extract module path before ".lora_",
// prepend "decoder." if needed, append ".weight".
static std::string lora_base_name(const std::string & key) {
    std::string s = key;

    // Strip each wrapper, including repeated PEFT exports.
    static const char * prefixes[] = {
        "base_model.model.",  // PEFT
        "diffusion_model.",   // ComfyUI official ACE-Step format
    };
    bool stripped;
    do {
        stripped = false;
        for (const char * pfx : prefixes) {
            const size_t len = strlen(pfx);
            if (s.compare(0, len, pfx) == 0) {
                s.erase(0, len);
                stripped = true;
                break;
            }
        }
    } while (stripped);

    // strip known PEFT module wrapper infixes.
    // HuggingFace PEFT wraps modules as e.g. "mlp.original_mlp.proj" instead
    // of "mlp.proj".  The ".original_<module>." segment is redundant.
    static const char * peft_infixes[] = {
        ".original_mlp.",
        ".original_self_attn.",
        ".original_cross_attn.",
    };
    for (const char * infix : peft_infixes) {
        size_t ipos = s.find(infix);
        if (ipos != std::string::npos) {
            s.erase(ipos, strlen(infix) - 1);  // erase infix, keep trailing dot
            break;
        }
    }

    // everything before ".lora_" is the module path
    size_t pos = s.find(".lora_");
    if (pos == std::string::npos) {
        return "";
    }
    s = s.substr(0, pos);

    // ensure decoder prefix (PEFT wraps the decoder directly,
    // so the internal path starts at "layers." not "decoder.layers.")
    if (s.compare(0, 8, "decoder.") != 0) {
        s = "decoder." + s;
    }

    return s + ".weight";
}
