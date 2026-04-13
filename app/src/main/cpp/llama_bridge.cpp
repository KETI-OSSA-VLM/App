#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define LOG_TAG "ACCV_LLAMA"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model*   g_model      = nullptr;
static llama_context* g_ctx        = nullptr;
static mtmd_context*  g_ctx_vision = nullptr;
static llama_pos      g_n_past_after_prefill = 0;
static llama_token    g_first_gen_token = -1;  // first token generated after prefill, for T1 logit restore

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_genionputtest_llamacpp_LlamaCppBridge_loadModel(
        JNIEnv* env,
        jobject,
        jstring modelPathJ,
        jstring mmprojPathJ,
        jint    nThreads
) {
    const char* model_path  = env->GetStringUTFChars(modelPathJ,  nullptr);
    const char* mmproj_path = env->GetStringUTFChars(mmprojPathJ, nullptr);

    // Clean up any previously loaded model
    if (g_ctx_vision) { mtmd_free(g_ctx_vision);        g_ctx_vision = nullptr; }
    if (g_ctx)        { llama_free(g_ctx);               g_ctx        = nullptr; }
    if (g_model)      { llama_model_free(g_model);       g_model      = nullptr; }

    // Load LLM
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only: GGML_VULKAN=OFF build, no GPU backend available
    g_model = llama_load_model_from_file(model_path, mparams);
    if (!g_model) {
        LOGE("Failed to load model: %s", model_path);
        env->ReleaseStringUTFChars(modelPathJ, model_path);
        env->ReleaseStringUTFChars(mmprojPathJ, mmproj_path);
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = 4096;
    cparams.n_threads      = (uint32_t)nThreads;
    cparams.n_threads_batch= (uint32_t)nThreads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // faster inference + lower memory (required for q8_0 KV cache)
    cparams.type_k         = GGML_TYPE_Q8_0; // 8-bit KV cache: ~50% memory vs f16, minimal quality loss
    cparams.type_v         = GGML_TYPE_Q8_0;
    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model); g_model = nullptr;
        env->ReleaseStringUTFChars(modelPathJ, model_path);
        env->ReleaseStringUTFChars(mmprojPathJ, mmproj_path);
        return JNI_FALSE;
    }

    // Load vision (mmproj)
    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu       = false; // vision encoder on CPU (Vulkan overhead is slower for ViT)
    vparams.n_threads     = (int32_t)nThreads;
    vparams.print_timings = true;
    g_ctx_vision = mtmd_init_from_file(mmproj_path, g_model, vparams);
    if (!g_ctx_vision) {
        LOGE("Failed to load mmproj: %s", mmproj_path);
        llama_free(g_ctx);         g_ctx   = nullptr;
        llama_model_free(g_model); g_model = nullptr;
        env->ReleaseStringUTFChars(modelPathJ, model_path);
        env->ReleaseStringUTFChars(mmprojPathJ, mmproj_path);
        return JNI_FALSE;
    }

    LOGI("Model ready: %s", model_path);
    env->ReleaseStringUTFChars(modelPathJ, model_path);
    env->ReleaseStringUTFChars(mmprojPathJ, mmproj_path);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_example_genionputtest_llamacpp_LlamaCppBridge_generate(
        JNIEnv* env,
        jobject,
        jstring imagePathJ,
        jstring promptTextJ,
        jint    maxNewTokens
) {
    if (!g_model || !g_ctx || !g_ctx_vision) {
        return env->NewStringUTF("ERROR: Model not loaded.");
    }

    const char* image_path = env->GetStringUTFChars(imagePathJ,  nullptr);
    const char* prompt     = env->GetStringUTFChars(promptTextJ, nullptr);

    g_n_past_after_prefill = 0;
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Load image bitmap
    LOGI("Loading image: %s", image_path);
    mtmd_bitmap* bitmap = mtmd_helper_bitmap_init_from_file(g_ctx_vision, image_path);
    if (!bitmap) {
        LOGE("Failed to load image: %s", image_path);
        env->ReleaseStringUTFChars(imagePathJ,  image_path);
        env->ReleaseStringUTFChars(promptTextJ, prompt);
        return env->NewStringUTF("ERROR: Failed to load image.");
    }
    LOGI("Image loaded OK. nx=%u ny=%u", mtmd_bitmap_get_nx(bitmap), mtmd_bitmap_get_ny(bitmap));

    // Build prompt with image marker
    std::string marker(mtmd_default_marker());
    std::string prompt_str =
        "<|im_start|>system\n"
        "You are analyzing a CCTV surveillance scene. Focus on people and their actions: what they are doing, how they are moving, and any notable events. Only describe what is clearly visible. Be concise.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n" + marker + "\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    LOGI("Prompt: %s", prompt_str.c_str());

    // Tokenize text + image
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text = { prompt_str.c_str(), /*add_special=*/true, /*parse_special=*/true };
    const mtmd_bitmap* bitmaps[] = { bitmap };
    LOGI("Tokenizing...");
    int32_t tok_ret = mtmd_tokenize(g_ctx_vision, chunks, &input_text, bitmaps, 1);
    if (tok_ret != 0) {
        LOGE("mtmd_tokenize failed: %d", tok_ret);
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        env->ReleaseStringUTFChars(imagePathJ,  image_path);
        env->ReleaseStringUTFChars(promptTextJ, prompt);
        return env->NewStringUTF("ERROR: Tokenize failed.");
    }
    LOGI("Tokenize OK. n_chunks=%zu n_tokens=%zu", mtmd_input_chunks_size(chunks), mtmd_helper_get_n_tokens(chunks));

    // Eval (prefill: text tokens + image encoding)
    llama_pos n_past = 0;
    LOGI("Starting eval (image encode + prefill)...");
    int32_t eval_ret = mtmd_helper_eval_chunks(
        g_ctx_vision, g_ctx, chunks,
        /*n_past=*/0, /*seq_id=*/0, /*n_batch=*/512, /*logits_last=*/true, &n_past
    );
    mtmd_bitmap_free(bitmap);
    mtmd_input_chunks_free(chunks);

    if (eval_ret != 0) {
        LOGE("mtmd_helper_eval_chunks failed: %d", eval_ret);
        env->ReleaseStringUTFChars(imagePathJ,  image_path);
        env->ReleaseStringUTFChars(promptTextJ, prompt);
        return env->NewStringUTF("ERROR: Eval failed.");
    }
    g_n_past_after_prefill = n_past;
    LOGI("Eval OK. n_past=%d. Saved g_n_past_after_prefill=%d. Starting generation...", (int)n_past, (int)g_n_past_after_prefill);

    // Sampler: min-p filters low-probability tokens, higher penalty reduces repetition/drift
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.2f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    llama_token eos = llama_vocab_eos(vocab);

    std::string result;

    g_first_gen_token = -1;
    for (int i = 0; i < maxNewTokens; i++) {
        llama_token token = llama_sampler_sample(sampler, g_ctx, -1);
        if (token == eos) {
            LOGI("EOS at token %d", i);
            break;
        }

        // Save the first generated token so generateOnly() can restore logits
        if (i == 0) g_first_gen_token = token;

        char piece[256];
        int32_t len = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (len > 0) result.append(piece, len);

        if (i % 10 == 0) LOGI("Generated %d tokens so far: %s", i, result.c_str());

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("llama_decode failed at token %d", i);
            break;
        }
        n_past++;

        // Stop after first sentence (sentence-level stop for short-answer prompts)
        if (result.size() > 20) {
            char last = result.back();
            if (last == '.' || last == '!' || last == '?') {
                LOGI("Sentence end at token %d", i);
                break;
            }
        }
    }
    LOGI("Generation done. Total tokens: %zu", result.size());

    llama_sampler_free(sampler);

    env->ReleaseStringUTFChars(imagePathJ,  image_path);
    env->ReleaseStringUTFChars(promptTextJ, prompt);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_genionputtest_llamacpp_LlamaCppBridge_freeModel(
        JNIEnv*, jobject
) {
    if (g_ctx_vision) { mtmd_free(g_ctx_vision);        g_ctx_vision = nullptr; }
    if (g_ctx)        { llama_free(g_ctx);               g_ctx        = nullptr; }
    if (g_model)      { llama_model_free(g_model);       g_model      = nullptr; }
    LOGI("Model freed.");
}

JNIEXPORT jstring JNICALL
Java_com_example_genionputtest_llamacpp_LlamaCppBridge_generateOnly(
        JNIEnv* env,
        jobject,
        jint maxNewTokens,
        jstring hintJ
) {
    if (!g_model || !g_ctx || !g_ctx_vision) {
        return env->NewStringUTF("ERROR: Model not loaded.");
    }
    if (g_n_past_after_prefill == 0) {
        return env->NewStringUTF("ERROR: No prefill state available.");
    }
    if (g_first_gen_token < 0) {
        return env->NewStringUTF("ERROR: No first-gen token saved.");
    }

    const char* hint_cstr = env->GetStringUTFChars(hintJ, nullptr);
    std::string hint(hint_cstr);
    env->ReleaseStringUTFChars(hintJ, hint_cstr);

    // Remove tokens generated since prefill, keeping image+prompt KV cache intact.
    llama_memory_seq_rm(llama_get_memory(g_ctx), 0, g_n_past_after_prefill, INT32_MAX);
    llama_pos n_past = g_n_past_after_prefill;

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.2f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.2f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

    llama_token eos = llama_vocab_eos(vocab);
    std::string result;

    if (!hint.empty()) {
        // Hint path: tokenize hint and decode into KV cache (replaces prime decode)
        LOGI("generateOnly: hint='%s'", hint.c_str());
        std::vector<llama_token> hint_tokens(hint.size() + 16);
        int32_t n_hint = llama_tokenize(
            vocab, hint.c_str(), (int32_t)hint.size(),
            hint_tokens.data(), (int32_t)hint_tokens.size(),
            /*add_special=*/false, /*parse_special=*/false
        );
        if (n_hint > 0) {
            hint_tokens.resize(n_hint);
            llama_batch hint_batch = llama_batch_get_one(hint_tokens.data(), n_hint);
            if (llama_decode(g_ctx, hint_batch) != 0) {
                LOGE("generateOnly: hint decode failed — falling back to prime decode");
                llama_batch prime = llama_batch_get_one(&g_first_gen_token, 1);
                if (llama_decode(g_ctx, prime) != 0) {
                    llama_sampler_free(sampler);
                    return env->NewStringUTF("ERROR: prime decode failed.");
                }
                n_past++;
                char piece[256];
                int32_t len = llama_token_to_piece(vocab, g_first_gen_token, piece, sizeof(piece), 0, true);
                if (len > 0) result.append(piece, len);
            } else {
                n_past += n_hint;
                LOGI("generateOnly: hint decoded OK, n_past=%d", (int)n_past);
            }
        } else {
            LOGE("generateOnly: hint tokenize failed, using prime decode");
            llama_batch prime = llama_batch_get_one(&g_first_gen_token, 1);
            if (llama_decode(g_ctx, prime) != 0) {
                llama_sampler_free(sampler);
                return env->NewStringUTF("ERROR: prime decode failed.");
            }
            n_past++;
            char piece[256];
            int32_t len = llama_token_to_piece(vocab, g_first_gen_token, piece, sizeof(piece), 0, true);
            if (len > 0) result.append(piece, len);
        }
    } else {
        // No hint: original prime decode path
        llama_batch prime = llama_batch_get_one(&g_first_gen_token, 1);
        if (llama_decode(g_ctx, prime) != 0) {
            LOGE("generateOnly: prime decode failed");
            llama_sampler_free(sampler);
            return env->NewStringUTF("ERROR: prime decode failed.");
        }
        n_past++;
        char piece[256];
        int32_t len = llama_token_to_piece(vocab, g_first_gen_token, piece, sizeof(piece), 0, true);
        if (len > 0) result.append(piece, len);
    }

    // Decoding loop
    for (int i = 0; i < maxNewTokens; i++) {
        llama_token token = llama_sampler_sample(sampler, g_ctx, -1);
        if (token == eos) {
            LOGI("generateOnly: EOS at token %d", i);
            break;
        }

        char piece[256];
        int32_t len = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (len > 0) result.append(piece, len);

        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("generateOnly: llama_decode failed at token %d", i);
            llama_sampler_free(sampler);
            return env->NewStringUTF("ERROR: decode failed.");
        }
        n_past++;

        if (result.size() > 20) {
            char last = result.back();
            if (last == '.' || last == '!' || last == '?') {
                LOGI("generateOnly: sentence end at token %d", i);
                break;
            }
        }
    }

    llama_sampler_free(sampler);
    LOGI("generateOnly done. hint='%s' result='%s'", hint.c_str(), result.c_str());
    return env->NewStringUTF(result.c_str());
}

} // extern "C"
