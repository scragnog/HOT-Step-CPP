/**
 * generate_translations.mjs — Translate en.json to target locales
 * Uses Google Translate free API via fetch (no package needed).
 * Run: node generate_translations.mjs
 */
import { readFileSync, writeFileSync } from 'fs';

const LOCALES = ['ru', 'zh-CN', 'ja', 'ko'];
const LOCALE_FILES = { 'ru': 'ru', 'zh-CN': 'zh', 'ja': 'ja', 'ko': 'ko' };

// Technical terms that should NOT be translated
const KEEP_ENGLISH = new Set([
  'DiT', 'VAE', 'LM', 'LoRA', 'LoKR', 'GGUF', 'BPM', 'APG', 'CFG', 'CFG++',
  'DPM++', 'DDIM', 'SGM', 'ODE', 'NFE', 'RK4', 'CoT', 'HSLAT', 'PP-VAE',
  'VST', 'VST3', 'VRAM', 'Opus', 'FLAC', 'WAV', 'MP3', 'RMS', 'DCW',
  'Euler', 'Heun', 'HOT-Step', 'HOT-Step CPP', 'Genius', 'OpenAI', 'Gemini',
  'Anthropic', 'Claude', 'GPT', 'Ollama', 'LM Studio', 'Unsloth',
  'safetensors', '.safetensors', '.latent', 'HuggingFace',
  'NoFSQ', 'FSQ', 'SNR-t', 'Log-SNR',
]);

async function translate(text, targetLang) {
  // Skip empty or very short strings
  if (!text || text.length < 2) return text;
  // Skip if it's a technical placeholder
  if (text.startsWith('{{') || text.startsWith('NO USER INPUT')) return text;
  
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=${targetLang}&dt=t&q=${encodeURIComponent(text)}`;
  
  try {
    const res = await fetch(url);
    const data = await res.json();
    // Response is nested arrays: [[["translated","original",null,null,x],...]]
    let result = '';
    if (Array.isArray(data) && Array.isArray(data[0])) {
      for (const segment of data[0]) {
        if (segment[0]) result += segment[0];
      }
    }
    return result || text;
  } catch (err) {
    console.error(`  ⚠ Failed to translate "${text.substring(0, 40)}..." to ${targetLang}:`, err.message);
    return text; // fallback to English
  }
}

// Rate limiter — Google blocks if we go too fast
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  const en = JSON.parse(readFileSync('./src/i18n/locales/en.json', 'utf-8'));
  const entries = Object.entries(en.translation);
  
  for (const locale of LOCALES) {
    const outFile = LOCALE_FILES[locale];
    console.log(`\n🌍 Translating to ${locale} (${entries.length} keys)...`);
    
    const translated = {};
    let done = 0;
    
    for (const [key, value] of entries) {
      done++;
      if (done % 50 === 0) console.log(`  ${done}/${entries.length}...`);
      
      const result = await translate(value, locale);
      translated[key] = result;
      
      // Small delay to avoid rate limiting (50ms between requests)
      await sleep(50);
    }
    
    const output = { translation: translated };
    writeFileSync(`./src/i18n/locales/${outFile}.json`, JSON.stringify(output, null, 2) + '\n', 'utf-8');
    console.log(`✅ Wrote ${outFile}.json (${Object.keys(translated).length} keys)`);
  }
  
  console.log('\n🎉 All translations complete!');
}

main().catch(console.error);
