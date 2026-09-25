# Assistant

A chat sidebar backed by an LLM you configure, with a live snapshot of your current settings on every message. Ask it to explain a control, recommend settings for a genre, troubleshoot a bad-sounding render, or write and rewrite lyrics and the style caption, then apply what it suggests with one click.

![Assistant](../../images/studio-assistant.webp)
<!-- screenshot: needed -->

## Where to find it

The "Assistant" toggle in the sidebar, which opens a resizable chat panel on the right side of the window.

It needs at least one working LLM provider configured under Settings > AI Services: Gemini, OpenAI, Anthropic, Ollama, LM Studio, Unsloth, llama.cpp, or a custom OpenAI-compatible endpoint. This is the same provider registry Lyric Studio uses, so a provider set up for one works for both. Cloud providers (Gemini, OpenAI, Anthropic) need an API key in the "API Keys" section; local providers (Ollama, LM Studio, Unsloth, llama.cpp, OpenAI-compatible) need their base URL, and some take an optional key or username/password. The Provider dropdown only lists providers the app can currently reach, and marks the rest "(unavailable)".

## Workflow

1. Open the Assistant from the sidebar.
2. Pick a provider and model from the two dropdowns at the top. The first available provider is auto-selected the first time; after that your choice is remembered.
3. Type a question or request and press Enter (Shift+Enter for a new line), or click one of the starter suggestions on the welcome screen ("Set me up for lo-fi hip hop", "What solver should I use for clean vocals?", "Review my current settings and suggest improvements", "How do I reduce metallic artifacts?").
4. The response streams in. If the model produces a reasoning block, it appears above the answer as a collapsible "Thought process" section.
5. If the response includes setting changes, a "Suggested Changes" card lists each one as a from-to row. Apply a single row or click "Apply All"; applied rows get a checkmark.
6. Applying a change takes effect immediately, either as an engine parameter (visible right away in Create) or as a content field (caption, lyrics, and the rest, written to the same storage Create reads from).

## Controls

| Control | What it does |
|---|---|
| Provider | Which configured LLM answers. Unavailable providers are shown disabled. |
| Model | Which model of the selected provider to use. Defaults to the provider's default model. |
| Message input | Ask a question or request a change. Enter sends, Shift+Enter adds a line break. |
| Clear chat (trash icon) | Discards the conversation shown in the panel and cancels a response that's still streaming. |
| Suggested Changes card | Appears under a response that included at least one setting change. Shows the current value, the proposed value, and an Apply button per row, plus Apply All. |
| Thought process | Collapsible block showing the model's reasoning, when the model provides one. Auto-expanded while still streaming. |

## Tips and limits

The assistant can only change what its knowledge base lists as a settable field: models, adapters, solver, scheduler, guidance mode and its sub-parameters, inference steps, shift, seed, batch size, LM settings, all of post-processing (spectral lifter, denoiser, mastering, PP-VAE, StableStep, DCW), latent shift and rescale, duration/auto-trim, and the content fields caption, lyrics, instrumental, bpm, duration, keyScale, timeSignature and vocalLanguage. It cannot start a generation, open the library, install a model, or navigate the UI for you; it only describes, recommends, and sets values you then act on.

Every message resends the whole conversation so far plus a fresh JSON snapshot of your settings. Nothing is kept on the server between requests, and closing the panel or reloading the page drops the conversation; there's no chat history to come back to.

The settings snapshot's content fields (caption, lyrics, and the rest) are read from the same storage the Create panel writes to. If you're in a different studio, the assistant still sees the mode you're in, but suggested content edits apply to Create's fields, not that studio's own inputs.

<!-- TODO(verify): whether the assistant validates its own lyric edits against the app's section-label and parenthesis rules (only the app's recognized section tags; parentheses are sung as backing vocals) before offering them, or just follows the instructions in its knowledge base and can still get it wrong. -->

If a generation or stem job is using the GPU and a local provider needs it too, a chat request can time out. Raise the "LLM call timeout" value in Settings > AI Services (the assistant's own timeout error names an "Environment" tab; the control is actually in AI Services).

## Related

- [Settings](settings.md)
- [Lyric Studio](lyric-studio.md)
- [Generation](../generation.md)
- [Quality](../quality.md)
