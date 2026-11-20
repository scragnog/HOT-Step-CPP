export interface ProviderInfo {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
  /** True when the model runs on the user's own machine (Ollama, LM Studio,
   *  llama.cpp, a self-hosted OpenAI-compatible endpoint). The UI used to infer
   *  this by testing `id === 'gemini'` and calling everything else cloud, which
   *  labelled a local LM Studio server as cloud and had users asking whether
   *  local models were supported at all. Declared by the provider instead. */
  local: boolean;
}

