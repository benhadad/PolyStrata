# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines, released on 2026-11-20: a mixture of 128 experts, 8 used a token, in 24
layers. Layers 6 to 17 run three times for each token, so a token goes through 48 layers with the weights of 24. Five
layers in six look back only 1,024 tokens. PolyStrata runs [ggml-org's GGUF files](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
and [Unsloth's UD-Q2_K_XL](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) on the strata-poly engine.

**It is experimental and new.** Support was added on 2026-11-21. It has run on one PC so far (below). Its logits were
checked against llama.cpp's on the same files (below): the same most likely next token after each of the prompts, and
the same at 29 to 32 of 32 steps with llama.cpp's own answer fed in.

## How fast is it?

> **Measured on:** RTX 2070 Super 8 GB, Ryzen 5 3600XT, 62 GB DDR4, Ubuntu 22.04, one request, warm file cache, the
> server's flags (128K context, q8_0 caches).

| | IQ4_NL prompt | IQ4_NL writing | UD-Q2_K_XL prompt | UD-Q2_K_XL writing |
| --- | ---: | ---: | ---: | ---: |
| 302-token prompt | 270 tokens/s | 37.4 tokens/s | 366 tokens/s | 46.9 tokens/s |
| 1,500-token prompt | 434 tokens/s | 36.7 tokens/s | 544 tokens/s | 45.2 tokens/s |
| 34,800-token prompt (IQ4_NL, through the server) | 232 tokens/s | 28.1 tokens/s | | |

IQ4_NL's speed depends on its expert profile (`--expert-profile`, which the server keeps up to date): it chooses the
experts the card holds. On 100 questions (50 GSM8K, 50 MMLU-Pro, greedy, medium reasoning) IQ4_NL got 49 and 44
right, UD-Q2_K_XL 48 and 41; the two gave the same answer to 92 of the 100.

About 4.6 GiB of experts fit on the 8 GB card. At most 7.5 GiB of VRAM was in use (nvidia-smi, IQ4_NL, a
31,600-token prompt). The engine's RAM was 16.5 GiB with IQ4_NL.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** card. About 3.1 GiB of VRAM before any expert at an 8K context (measured on an 8 GB card), about 4 GiB at 32K. |
| **RAM** | About **22 GB** (IQ4_NL), **14 GB** (UD-Q2_K_XL) or **32 GB** (Q8_0) beside an 8 GB card, setup's estimate (the engine itself used 16.5 GiB with IQ4_NL, 9.1 GiB with UD-Q2_K_XL). The experts the card does not hold stay in RAM. Beside a 24 GB card the IQ4_NL file's experts fit in VRAM by the same estimate. |
| **Disk** | 17.6, 9.1 or 27.8 GB for the model file. |
| **System** | Linux and a current NVIDIA driver. Setup asks before it installs it on Windows: it was not built or run there. |

It is not offered on AMD or Intel cards or on a Mac: the engine code for it was written for NVIDIA cards only. For now
it also runs on one card (no layer split) and answers one request at a time (no `--parallel`); setup leaves those
options out for it. Its attention caches are fp16, or q8_0 with `--kv q8_0`, which halves them: a 128K context then
takes about 2.3 GiB of VRAM when it is full, in fp16 4.3 GiB. On an 8 GB card, use q8_0 for contexts past 32K. The file allows
1,048,576 tokens and the engine sets no limit of its own, but the cache stays in VRAM (about 17 KiB a token in q8_0;
the engine's KV streaming is not wired up for this model), and it was tried up to a 128K context, with prompts of up
to 35K tokens. It reads text only for now. The repository has
an image encoder too, which setup does not download. It has no draft model, so it writes one token at a time.

## Install

```bash
./setup.sh --family inkling
```

```bash
./setup.sh --family inkling --model UD-Q2_K_XL
```

```bash
./setup.sh --family inkling --model Q8_0
```

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card (estimate) |
| --- | --- | ---: | ---: |
| **IQ4_NL** | 4-bit experts in layers 0 to 19 (with an importance matrix), 8-bit everywhere else | 17.6 GB | about 22 GB |
| **UD-Q2_K_XL** | Unsloth's: 2-bit experts (Q2_K, IQ1_S and Q2_0), higher where it matters. The fastest, a little less accurate | 9.1 GB | about 14 GB |
| **Q8_0** | 8-bit throughout: the closest to the model as trained | 27.8 GB | about 32 GB |

Setup takes IQ4_NL unless you name another size. UD-Q2_K_XL comes from Unsloth's repository in the architecture's
native form (`inkling2`, the loop not unrolled); the engine reads that form as it is. Q8_0 was not run. ggml-org's model card gives the KL divergence against the BF16 file on
wikitext-2: 0.0030 for Q8_0 and 0.0095 for IQ4_NL. PolyStrata did not measure that.

## How it talks

The server reads these from the tokenizer that setup writes from the GGUF (`pre` = `inkling2`):

- **Tokens.** Byte-level BPE with 200,058 tokens, llama.cpp's `inkling2` split (its GPT-4o pattern) and, as in the
  model's `tokenizer.json`, `ignore_merges`: a piece that is a token as a whole is that token. No token starts a
  prompt. A reply ends at `<|content_model_end_sampling|>` (the GGUF's end of text) or at `<|endoftext|>`.
- **Reply.** A reply is a list of blocks. Each starts with `<|message_model|>` and ends with `<|end_message|>`. A
  `<|content_thinking|>` block is the reasoning, a `<|content_text|>` block is the answer, and a tool call is
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`. This is the form
  llama.cpp reads (its Inkling parser, ggml-org/llama.cpp PR 32381). The model writes the first header itself.
- **Reasoning effort.** The template writes "Thinking effort level" in a system block. `reasoning_effort` low, medium
  and high give 0.2, 0.7 and 0.9. High (0.9) is the default. Thinking turned off gives 0. The model can still think a
  little at 0: its model card says the lowest effort does not turn reasoning off. "xhigh" and "max" are 0.9 too,
  and "minimal" is thinking off (0), as for every model here: the template's 0.1 and 0.99 are not offered.
- **Sampling.** Setup writes the GGUF's own defaults into the run config: temperature 1.0, top_k 50, top_p 1.0. A
  `"sampling"` block you wrote yourself is kept.
- **Forced calls.** `tool_choice` naming one tool is written for the model after its thinking, or at the start of
  the reply with thinking off. The reply ends with that call: a block the model opens after it is not sent.
  `"required"` with several tools is left to the model.

## How it was checked

Token ids against Hugging Face's `tokenizers` library on the model's `tokenizer.json`, the reference. Prompts and
replies against llama.cpp 29b52d398b, the commit that added the model, built on the CPU and run without the model's
weights. 51 of the strings (53 cases), the 14 prompts and the 11 replies are kept in `serve/fixtures/` and checked by
`serve/test_inkling.py`.

- **Tokens.** 1,006 strings, about 587,000 tokens: English, Japanese, Chinese, Korean, Arabic, Hindi and Russian text,
  code, emoji, whitespace runs, control tokens, PolyStrata's docs and source in pieces, and 300 random strings. Every
  id was the same as Hugging Face's. llama.cpp's differed in 11 of the strings (see below).
- **Prompts.** 14 requests rendered by llama.cpp's chat code and by the server were the same text: effort levels, a
  system prompt, tools, two tool calls and their answers, earlier reasoning, Unicode and quotes.
- **Replies.** 11 replies (llama.cpp's own Inkling tests and a few more) gave the same reasoning, answer and tool calls,
  fed whole, in 5-character pieces and one character at a time.

The engine's output was checked against llama.cpp's on the same files, IQ4_NL and UD-Q2_K_XL, for a short prompt and prompts
of 302 and 1,500 tokens, each run twice: the logits' cosine similarity was 0.9989 to 0.9998, and 16 to 20 of the 20
most likely tokens were the same. With llama.cpp's own 32-token answer fed in one token at a time, the most likely
token was the same at 29 to 32 of the 32 steps. Where it differed, llama.cpp's two best tokens were within 0.14 of each
other in logit, except at one step (UD-Q2_K_XL, the 1,500-token prompt, in every run), where they were 1.0 apart. Two
runs of the same input do not give identical logits here (cosine 0.9992 between two runs of the 302-token prompt).

## Known differences

- llama.cpp 29b52d398b does not set `ignore_merges` for `inkling2`, so a word that is one token but that no chain of
  merges reaches comes out as several there: `kwargs` is one token here and with Hugging Face, three in llama.cpp.
  That changed 11 of the 1,006 strings above. PolyStrata follows Hugging Face. `serve/fixtures/inkling2_golden.json`
  keeps llama.cpp's ids for one of them (`kwargs`).
- A message whose content is a list of text parts is one block here and one block a part in llama.cpp. The server
  joins the parts for every model.
- llama.cpp checks a tool call against the declared tools with a grammar while the model writes it. PolyStrata has no
  grammar: it reads the call when its block is closed. A call to a tool that was not declared is shown as text.

## How does it work?

The engine side is in `src/poly/inkling.cpp`. The server side is `InklingParser` in `serve/frontend.py`, and the tests
are `serve/test_inkling.py` and `tools/test_setup_inkling.py`.
