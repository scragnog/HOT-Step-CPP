import { Download, Loader2, Play, RotateCcw, Save, Upload, X } from 'lucide-react';
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { HeartmulaPreviewListResponse, HeartmulaPreviewSettings, HeartmulaRecipe, HeartmulaStatusResponse, HeartmulaTrainRequest } from '../../../../server/src/contracts/heartmulaTraining';
import { HEARTMULA_RECIPE_LEGACY, heartmulaDefaultSteps } from '../../../../server/src/contracts/heartmulaTraining';
import { useTrainingStore } from '../../stores/trainingStore';
import { TrainingChart } from './TrainingChart';
import { formatDuration, heartmulaEta } from './heartmulaEta';
import { PreviewField, TrainingPreviewSettings, TrainingRunPicker, TrainingPresetChoice } from './TrainingPreviewPanel';
import { TrainingBatchCard, TrainingAllStagesCard, TrainingRunPanel } from './Yue2StageCard';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { Yue2AitkBatchWizard } from './Yue2AitkBatchWizard';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

export interface HeartmulaTrainCardProps {
  datasetId?: string;
  status: HeartmulaStatusResponse;
  slug: string;
  busy: boolean;
  /** The current run's preview takes (the controller's query), for the ETA. */
  previews?: HeartmulaPreviewListResponse | null;
  /** True after the trainer's start event, when control files are accepted. */
  trainStarted: boolean;
  blockedReason?: string;
  onTrain: (request: HeartmulaTrainRequest) => void;
  onChain?: (request: HeartmulaTrainRequest, datasetIds?: string[]) => void;
  onCheckpoint: (run: string) => void;
  onCancel: () => void;
}

const BUTTON = 'px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 flex items-center gap-2';
const INPUT = 'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50';
type NumericRecipeKey = Exclude<keyof HeartmulaRecipe, 'trainMlp' | 'variant' | 'quant' | 'cropSeconds' | 'eos'>;
const FIELDS: Array<{ key: NumericRecipeKey; label: string; info: string; min: number; max: number; step: number }> = [
  { key: 'steps', label: 'Updates', info: 'Number of optimizer updates. More steps take longer and can overfit the dataset.', min: 1, max: 1_000_000, step: 1 },
  { key: 'rank', label: 'Rank', info: 'Adapter capacity. Higher ranks use more memory. The starting recipe uses rank 64.', min: 1, max: 256, step: 1 },
  { key: 'alpha', label: 'Alpha', info: 'Adapter scaling relative to rank. The starting recipe pairs alpha 64 with rank 64.', min: 0.001, max: 1024, step: 0.001 },
  { key: 'learningRate', label: 'Learning rate', info: 'Size of each weight update. Larger values learn faster but can make training unstable.', min: 1e-8, max: 0.1, step: 0.00001 },
  { key: 'gradAccum', label: 'Songs per update', info: 'Micro-batches per optimizer update. Higher values do more work per step without increasing the micro-batch size.', min: 1, max: 1024, step: 1 },
  { key: 'seed', label: 'Training seed', info: 'Seed for training randomness. Keep it fixed when comparing recipes.', min: 0, max: 0x7fffffff, step: 1 },
  { key: 'loraDropout', label: 'LoRA dropout', info: 'Randomly drops adapter inputs while training, which can reduce overfitting on a small dataset. 0 is off; 0.05 is a proven value that adds about 8% to each step and 1.3 GiB of memory.', min: 0, max: 0.9, step: 0.01 },
  { key: 'checkpointEvery', label: 'Save every', info: 'Save resumable state every this many steps. Zero saves only when stopping or finishing. Checkpoint previews, when on, render at every save.', min: 0, max: 1_000_000, step: 1 },
];

export function HeartmulaTrainCard({ datasetId, status, slug, busy, previews: takes, trainStarted, blockedReason, onTrain, onChain, onCheckpoint, onCancel }: HeartmulaTrainCardProps) {
  const { t } = useTranslation();
  const [edits, setEdits] = useState<Partial<HeartmulaRecipe>>({});
  const [batchOpen, setBatchOpen] = useState(false);
  const [batchIds, setBatchIds] = useState<string[] | null>(null);
  const [name, setName] = useState(slug);
  const [resumeRun, setResumeRun] = useState<string | null>(null);
  const [previewsOn, setPreviewsOn] = useState<boolean | null>(null);
  const [previewEdits, setPreviewEdits] = useState<Partial<HeartmulaPreviewSettings>>({});
  const [manualResume, setManualResume] = useState('');
  const [presetKey, setPresetKey] = useState('approved');
  const [presetName, setPresetName] = useState('');
  const [presetError, setPresetError] = useState('');
  const [cropDraft, setCropDraft] = useState(120);
  const fileRef = useRef<HTMLInputElement>(null);
  const validRecipe = (value: unknown): value is HeartmulaRecipe => !!value && typeof value === 'object' && FIELDS.every(field => {
    const number = (value as Record<string, unknown>)[field.key];
    return typeof number === 'number' && Number.isFinite(number) && number >= field.min && number <= field.max
      && (field.step !== 1 || Number.isInteger(number));
  });
  const [presets, setPresets] = useState<Array<{ name: string; recipe: HeartmulaRecipe }>>(() => {
    try {
      const stored: unknown = JSON.parse(localStorage.getItem('hs-heartmula-training-presets') ?? '[]');
      return Array.isArray(stored) ? stored.filter(item => typeof item?.name === 'string' && item.name.trim() && item.name.length <= 64 && validRecipe(item.recipe)).slice(0, 32).map(item => ({ name: item.name, recipe: { ...Object.fromEntries(FIELDS.map(field => [field.key, item.recipe[field.key]])), trainMlp: item.recipe.trainMlp === true, cropSeconds: Number(item.recipe.cropSeconds) || 0, ...(typeof item.recipe.variant === 'string' ? { variant: item.recipe.variant } : {}), ...(typeof item.recipe.quant === 'string' ? { quant: item.recipe.quant } : {}) } as unknown as HeartmulaRecipe })) : [];
    } catch { return []; }
  });
  const persist = (next: typeof presets) => {
    try { localStorage.setItem('hs-heartmula-training-presets', JSON.stringify(next)); setPresets(next); setPresetError(''); }
    catch { setPresetError(t('trainingStudio.heartmula.presetStorageError', 'Could not save presets in this browser.')); }
  };
  const manualPath = manualResume.trim().replaceAll('\\', '/').replace(/\/$/, '');
  const selectedResume = resumeRun ?? status.runs.find(item => item.run === manualPath || manualPath.endsWith('/' + item.run))?.run ?? null;
  const defaults = { ...status.defaults, steps: heartmulaDefaultSteps(status.prepare.preparedSamples, edits.gradAccum ?? status.defaults.gradAccum) };
  const recipe = selectedResume ? status.runs.find(run => run.run === selectedResume)?.recipe ?? status.defaults : { ...defaults, ...edits };
  // Previews render at every save; with saving only at the end, just the base and final takes.
  const cadence = (previewsOn ?? status.previewDefaults.everySteps > 0)
    ? (recipe.checkpointEvery > 0 ? recipe.checkpointEvery : recipe.steps) : 0;
  const previewSettings = { ...status.previewDefaults, ...previewEdits };
  const setPreview = (patch: Partial<HeartmulaPreviewSettings>) => setPreviewEdits(previous => ({ ...previous, ...patch }));
  const resetPreview = (key: keyof HeartmulaPreviewSettings) => () => setPreviewEdits(previous => { const next = { ...previous }; delete next[key]; return next; });
  const run = status.run;
  const [openedAt] = useState(Date.now);
  const showRun = run && (['queued', 'running', 'previewing', 'checkpointing'].includes(run.state)
    || run.updatedAt >= openedAt - 15 * 60_000);
  const ready = status.prepare.cacheValid && status.prepare.staleSamples === 0 && status.prepare.preparedSamples > 0;
  const resume = resumeRun !== null || manualResume.trim() !== '';
  const resumable = status.runs.some(item => item.run === selectedResume && item.resumable) && !busy;
  const jobLog = useTrainingStore(s => s.jobLog);
  // Time left: steps at the mean step time plus the previews still to come at their measured cost.
  const runLive = !!run && (run.state === 'running' || run.state === 'previewing' || run.state === 'checkpointing');
  const eta = runLive && run ? heartmulaEta({ step: run.step, totalSteps: run.totalSteps, msPerStep: run.msPerStep,
    everySteps: takes?.settings.everySteps ?? 0, previews: takes?.previews ?? [] }) : null;
  const job = useActiveTrainingJob();
  const ownLog = job?.kind === 'heartmula-train' && job.datasetId === datasetId ? jobLog : [];
  const locked = busy || resume;
  const edited = (key: keyof HeartmulaPreviewSettings) => !locked && previewEdits[key] !== undefined && previewEdits[key] !== status.previewDefaults[key];
  const cropOk = !recipe.cropSeconds || (Number.isFinite(recipe.cropSeconds) && recipe.cropSeconds >= 10 && recipe.cropSeconds <= 600);
  const valid = cropOk && FIELDS.every(field => Number.isFinite(recipe[field.key]) && recipe[field.key] >= field.min
    && recipe[field.key] <= field.max && (field.step !== 1 || Number.isInteger(recipe[field.key])))
    && Number.isInteger(cadence) && cadence >= 0 && cadence <= 1_000_000
    && Number.isInteger(previewSettings.seconds) && previewSettings.seconds >= 5 && previewSettings.seconds <= 240
    && Number.isInteger(previewSettings.seed) && previewSettings.seed >= 0
    && /^[a-z0-9][a-z0-9_-]{0,63}$/i.test(name);
  const numberField = (field: typeof FIELDS[number]) => <PreviewField key={field.key}
    label={t(`trainingStudio.heartmula.field.${field.key}`, field.label)}
    info={t(`trainingStudio.heartmula.field.${field.key}Info`, field.info)}
    onReset={!locked && edits[field.key] !== undefined && edits[field.key] !== defaults[field.key] ? () => setEdits(previous => {
      const next = { ...previous }; delete next[field.key]; return next;
    }) : undefined}>
    <input className={INPUT} type="number" min={field.min} max={field.max} step={field.step}
      value={Number.isFinite(recipe[field.key]) ? recipe[field.key] : ''} disabled={locked}
      onChange={event => setEdits(previous => ({ ...previous, [field.key]: event.target.valueAsNumber }))} />
  </PreviewField>;
  // Every recipe key, not just the number fields: the toggles, the crop and the base count as changes too.
  const recipeKeys: Array<keyof HeartmulaRecipe> = [...FIELDS.map(field => field.key), 'variant', 'quant', 'trainMlp', 'cropSeconds'];
  const changed = recipeKeys.filter(key => (recipe[key] ?? defaults[key]) !== defaults[key]).map(key => ({ key }));
  const shown = (key: keyof HeartmulaRecipe) => {
    const v = recipe[key];
    if (key === 'cropSeconds') return v ? `${v} s` : 'off';
    return typeof v === 'boolean' ? (v ? 'on' : 'off') : String(v);
  };
  // Saving presets never touches a run, so it stays open while one trains. During a run it saves
  // what that run is training with, which survives a page reload where the form does not.
  const liveRecipe = busy && status.run ? status.runs.find(item => item.run === status.run?.run)?.recipe : undefined;
  const presetRecipe: HeartmulaRecipe = liveRecipe ? { ...status.defaults, ...HEARTMULA_RECIPE_LEGACY, ...liveRecipe } : recipe;
  const savePreset = () => {
    if (!presetName.trim() || !validRecipe(presetRecipe)) return;
    persist([...presets.filter(item => item.name !== presetName.trim()), { name: presetName.trim(), recipe: { ...presetRecipe } }].slice(-32));
    setPresetName('');
  };
  const importPresets = async (file: File) => {
    try {
      if (file.size > 100_000) throw new Error('Preset file is too large');
      const data = JSON.parse(await file.text());
      if (data.version !== 1 || data.target !== 'heartmula' || !Array.isArray(data.presets) || data.presets.length > 32
        || !data.presets.every((item: { name?: unknown; recipe?: unknown }) => typeof item.name === 'string'
          && item.name.trim().length > 0 && item.name.length <= 64 && validRecipe(item.recipe))) throw new Error('Invalid HeartMuLa presets');
      const merged = new Map(presets.map(item => [item.name, item]));
      for (const item of data.presets) merged.set(item.name, { name: item.name, recipe: { ...Object.fromEntries(FIELDS.map(field => [field.key, item.recipe[field.key]])), trainMlp: item.recipe.trainMlp === true, cropSeconds: Number(item.recipe.cropSeconds) || 0, ...(typeof item.recipe.variant === 'string' ? { variant: item.recipe.variant } : {}), ...(typeof item.recipe.quant === 'string' ? { quant: item.recipe.quant } : {}) } as unknown as HeartmulaRecipe });
      persist([...merged.values()].slice(-32));
    } catch { setPresetError(t('trainingStudio.heartmula.presetImportError', 'Import a valid HeartMuLa preset file of up to 100 KB.')); }
  };
  const exportPreset = (preset: typeof presets[number]) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ version: 1, target: 'heartmula', presets: [preset] }, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'heartmula-training-preset.json'; link.click(); URL.revokeObjectURL(url);
  };
  // Keep the dataset-dependent budget omitted for API admission, including a batch prepared later.
  const requestRecipe: Partial<HeartmulaRecipe> = { ...recipe };
  if (edits.steps === undefined) delete requestRecipe.steps;
  const start = () => onTrain(selectedResume ? { resumeRun: selectedResume } : { adapterName: name, recipe: requestRecipe, preview: { ...previewEdits, everySteps: cadence } });
  const chain = () => onChain?.(selectedResume ? { resumeRun: selectedResume }
    : { ...(batchIds ? {} : { adapterName: name }), recipe: requestRecipe, preview: { ...previewEdits, everySteps: cadence } }, batchIds ?? undefined);
  return <div className="flex flex-col gap-4">
    <TrainingBatchCard disabled={busy || !onChain} onOpen={() => setBatchOpen(true)} description={t('trainingStudio.heartmula.batchInfo', 'Choose datasets, then apply the recipe below to each in order. The queue survives closing the page and server restarts.')} />
    {batchOpen && <Yue2AitkBatchWizard open family="heartmula" onClose={() => setBatchOpen(false)} onSelect={setBatchIds} />}
    {batchIds && <div className="flex items-center gap-3 text-xs text-amber-600 dark:text-amber-400">
      {t('trainingStudio.heartmula.batchSelected', 'Recipe applies to {{count}} datasets.', { count: batchIds.length })}
      <button type="button" disabled={busy} onClick={() => setBatchIds(null)}>{t('common.cancel', 'Cancel')}</button>
    </div>}
    <TrainingAllStagesCard disabled={busy || !onChain || !!blockedReason || (!ready && status.ready.encoderMissing) || (resume ? !resumable || !!batchIds : !valid)} onRun={chain}
      description={t('trainingStudio.heartmula.chainInfo', 'Prepares missing or stale HeartMuLa audio, then trains with the settings below. Unchanged prepared audio is reused.')} />
    <TrainingRunPanel title={t('trainingStudio.heartmula.selected', 'Backbone LoRA training is selected')}>
    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
      <div className="flex flex-col gap-1"><TrainingRunPicker label={t('trainingStudio.yue2.method.resumePrevious', 'Resume a previous run')}
        info={t('trainingStudio.heartmula.resumeInfo', 'Continue a saved run with its recipe, preview settings and optimizer state. Recipe edits apply only to a new run.')}
        value={resumeRun ?? ''} disabled={busy} onChange={value => { setResumeRun(value || null); setManualResume(''); }}
        options={[{ value: '', label: t('trainingStudio.yue2.method.resumeStartNew', 'Start a new run') },
          ...status.runs.map(item => ({ value: item.run, label: `${item.run} · ${item.step} · ${item.state}`, disabled: !item.resumable }))]} />
        {!resumeRun && <input className={`${INPUT} mt-2`} disabled={busy} value={manualResume} onChange={event => setManualResume(event.target.value)}
          placeholder={t('trainingStudio.heartmula.resumePath', 'Or paste a run folder (optional)')}
          title={t('trainingStudio.heartmula.resumePathInfo', 'Paste the ID or folder of a resumable run belonging to this dataset. Its saved recipe is restored.')} />}
        {resume && !resumable && !busy && <p role="alert" className="text-xs text-amber-600">{t('trainingStudio.heartmula.resumeInvalid', 'Choose a resumable run from this dataset.')}</p>}
      </div>
      <PreviewField label={t('trainingStudio.heartmula.adapterName', 'Adapter name')}
        info={t('trainingStudio.heartmula.adapterNameInfo', 'Name for the saved adapter. Use letters, numbers, underscores or hyphens, up to 64 characters.')}>
        <input className={INPUT} value={name} maxLength={64} disabled={locked} onChange={event => setName(event.target.value)} />
      </PreviewField>
    </div>
    <div className="mt-4">
      <ParamLabel label={t('trainingStudio.yue2.method.preset', 'Preset')} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
        info={t('trainingStudio.heartmula.defaultBudgetInfo', 'HOT-Approved caps the default budget at 30 epochs or 480 updates, with four songs per update and 0.05 dropout. Fast, Balanced and Thorough set explicit update budgets. Saved presets and update edits keep their values.')} />
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-2">
        {[{ key: 'approved', label: 'HOT-Approved', steps: status.defaults.steps, gradAccum: status.defaults.gradAccum },
          { key: 'fast', label: 'Fast', steps: 240, gradAccum: 4 }, { key: 'balanced', label: 'Balanced', steps: 480, gradAccum: 4 },
          { key: 'thorough', label: 'Thorough', steps: 720, gradAccum: 8 }].map(preset => <TrainingPresetChoice key={preset.key} disabled={locked}
            selected={presetKey === preset.key && FIELDS.every(field => recipe[field.key] === ({ ...status.defaults, steps: preset.steps, gradAccum: preset.gradAccum })[field.key])}
            onClick={() => { setPresetKey(preset.key); setEdits(preset.key === 'approved' ? {} : { ...status.defaults, steps: preset.steps, gradAccum: preset.gradAccum }); }}>
            <span className="text-base font-bold">{t(`trainingStudio.heartmula.preset.${preset.key}`, preset.label)}</span>
            <span className="text-xs text-zinc-500">{t('trainingStudio.heartmula.presetBudget', '{{steps}} × {{songs}} songs', { steps: preset.steps, songs: preset.gradAccum })}</span>
          </TrainingPresetChoice>)}
      </div>
      <div className="mt-3 flex items-center gap-2 flex-wrap">
        <ParamLabel label={t('trainingStudio.yue2.method.presets', 'Your presets')} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
          info={t('trainingStudio.heartmula.userPresetInfo', 'Save and share recipe settings. Presets never include dataset, adapter name or resume paths.')} />
        {presets.map(preset => <span key={preset.name} className={`inline-flex items-center gap-1 rounded-lg border-2 pl-3 pr-1 py-1 text-xs ${FIELDS.every(field => preset.recipe[field.key] === recipe[field.key])
          ? 'border-blue-500 bg-blue-500/15 text-blue-700 dark:text-blue-300' : 'border-zinc-300 dark:border-white/15 text-zinc-800 dark:text-zinc-100'}`}>
          <button type="button" disabled={locked} onClick={() => { setEdits({ ...preset.recipe }); setPresetKey('custom'); }} className="font-semibold hover:underline disabled:opacity-40">{preset.name}</button>
          <button type="button" disabled={!validRecipe(presetRecipe)} onClick={() => persist(presets.map(item => item.name === preset.name ? { ...item, recipe: { ...presetRecipe } } : item))}
            title={t('trainingStudio.yue2.method.presetUpdate', 'Overwrite this preset with the current settings')} className="rounded p-0.5 text-zinc-400 hover:text-amber-600 disabled:opacity-40"><Save size={12} /></button>
          <button type="button" onClick={() => exportPreset(preset)} title={t('trainingStudio.yue2.method.presetExport', 'Download this preset as a .json file to share or back up')} className="rounded p-0.5 text-zinc-400 hover:text-amber-600"><Download size={12} /></button>
          <button type="button" onClick={() => persist(presets.filter(item => item.name !== preset.name))} title={t('trainingStudio.yue2.method.presetRemove', 'Delete this preset')} className="rounded p-0.5 text-zinc-400 hover:text-red-600 disabled:opacity-40"><X size={12} /></button>
        </span>)}
        <input className={`${INPUT} !w-48 !py-1 !text-xs`} placeholder={t('trainingStudio.yue2.method.presetNamePlaceholder', 'New preset name')}
          maxLength={64} value={presetName} onChange={event => setPresetName(event.target.value)} />
        <button type="button" disabled={!presetName.trim() || !validRecipe(presetRecipe)} onClick={savePreset} className="px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">{t('trainingStudio.yue2.method.presetSave', 'Save current settings')}</button>
        <button type="button" onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10 disabled:opacity-40"><Upload size={12} />{t('trainingStudio.yue2.method.presetImport', 'Import')}</button>
        <input ref={fileRef} type="file" accept=".json,application/json" className="hidden" onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void importPresets(file); }} />
      </div>
      {presetError && <p role="alert" className="mt-1 text-xs text-red-500">{presetError}</p>}
      <div className="mt-3 flex items-center gap-2 flex-wrap">
        <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">{t('trainingStudio.yue2.method.changedSettings', 'Changed from defaults')}</span>
        {!changed.length ? <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.noChangedSettings', 'none: this form is the recipe defaults')}</span>
          : <>{changed.map(field => <span key={field.key} className="inline-flex items-center gap-1 rounded-lg border border-sky-500/40 bg-sky-500/10 pl-2 pr-1 py-0.5 text-[11px] text-sky-700 dark:text-sky-300">{field.key}: {shown(field.key)}
            <button type="button" disabled={locked} title={t('trainingStudio.yue2.method.resetSetting', 'Reset this setting to its default')} onClick={() => setEdits(previous => { const next = { ...previous }; delete next[field.key]; return next; })} className="rounded p-0.5 hover:text-sky-900 dark:hover:text-white disabled:opacity-40"><RotateCcw size={11} /></button>
          </span>)}<button type="button" disabled={locked} onClick={() => setEdits({})} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-[11px] font-semibold border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10 disabled:opacity-40"><RotateCcw size={11} />{t('trainingStudio.yue2.method.resetAll', 'Reset all to defaults')}</button></>}
      </div>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-3">
      <PreviewField label={t('trainingStudio.heartmula.baseVariant', 'Base model')}
        info={t('trainingStudio.heartmula.baseVariantInfo', 'The HeartMuLa variant the adapter is trained on. It plays natively on that variant and is re-based onto the others. Only variants with an installed global and local model are listed.')}>
        <StyledSelect accent="amber" className="w-full" disabled={locked} value={recipe.variant ?? '3B'}
          onChange={value => setEdits(previous => {
            const quants = status.bases?.find(b => b.variant === value)?.quants ?? [];
            return { ...previous, variant: value, quant: quants.includes(recipe.quant ?? 'F16') ? recipe.quant : quants[0] ?? 'F16' };
          })}
          options={(status.bases?.length ? status.bases : [{ variant: recipe.variant ?? '3B', quants: [] }]).map(b => ({ value: b.variant, label: b.variant }))} />
      </PreviewField>
      <PreviewField label={t('trainingStudio.heartmula.baseQuant', 'Base quant')}
        info={t('trainingStudio.heartmula.baseQuantInfo', 'The precision of the frozen base weights during training. F16 is the reference. Smaller quants use less VRAM and train on what you will generate with, but their rounding is baked into what the adapter learns.')}>
        <StyledSelect accent="amber" className="w-full" disabled={locked} value={recipe.quant ?? 'F16'}
          onChange={value => setEdits(previous => ({ ...previous, quant: value }))}
          options={(status.bases?.find(b => b.variant === (recipe.variant ?? '3B'))?.quants ?? [recipe.quant ?? 'F16']).map(q => ({ value: q, label: q }))} />
      </PreviewField>
      <PreviewField label={t('trainingStudio.yue2.method.trainingGpu', 'Training GPU')} info={t('trainingStudio.heartmula.gpuInfo', 'The current HeartMuLa trainer selects its CUDA device automatically. Per-run GPU selection is not supported.')}>
        <StyledSelect accent="amber" className="w-full" disabled value="auto" onChange={() => {}} options={[{ value: 'auto', label: t('trainingStudio.heartmula.gpuAuto', 'Automatic (CUDA)') }]} />
      </PreviewField>
      <PreviewField label={t('trainingStudio.yue2.method.adapterType', 'Adapter type')} info={t('trainingStudio.heartmula.adapterTypeInfo', 'The current HeartMuLa trainer supports backbone LoRA adapters.')}>
        <StyledSelect accent="amber" className="w-full" disabled value="lora" onChange={() => {}} options={[{ value: 'lora', label: 'LoRA' }]} />
      </PreviewField>
    </div>
    <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-3">
      {[FIELDS[0], FIELDS[7], FIELDS[5], FIELDS[1], FIELDS[2], FIELDS[3]].map(numberField)}
    </div>
    <details className="mt-4 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/50 dark:bg-black/10 p-3">
      <summary className="cursor-pointer text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.advanced', 'Advanced')}</summary>
      <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">{[FIELDS[6], FIELDS[4]].map(numberField)}</div>
      <Toggle className="mt-3" accent="amber" disabled={locked} checked={recipe.trainMlp === true}
        onChange={value => setEdits(previous => ({ ...previous, trainMlp: value }))}
        label={t('trainingStudio.heartmula.field.trainMlp', 'Train MLP layers')}
        info={t('trainingStudio.heartmula.field.trainMlpInfo', 'Also trains the MLP (feed-forward) layers of every backbone block, not only attention. The adapter is bigger, steps take longer and use more memory. Lets the MLP dial in the Adapters panel change the sound.')} />
      <Toggle className="mt-3" accent="amber" disabled={locked} checked={(recipe.cropSeconds ?? 0) > 0}
        onChange={value => setEdits(previous => ({ ...previous, cropSeconds: value ? cropDraft : 0 }))}
        label={t('trainingStudio.heartmula.field.crop', 'Crop songs')}
        info={t('trainingStudio.heartmula.field.cropInfo', 'Trains on the opening stretch of each song instead of the whole track, so starts stay right, and trains every fourth song whole so later sections and endings are still learned. Steps are faster on long songs; peak memory still follows the longest song, because whole songs are still trained. Songs shorter than the window always train whole.')} />
      {(recipe.cropSeconds ?? 0) > 0 && (
        <div className="mt-2 max-w-[16rem]">
          <PreviewField label={t('trainingStudio.heartmula.field.cropSeconds', 'Window (seconds)')}
            info={t('trainingStudio.heartmula.field.cropSecondsInfo', 'How much of the start of each song a cropped step trains on, 10 to 600 seconds. 120 is a good start.')}>
            <input className={INPUT} type="number" min={10} max={600} step={1} disabled={locked}
              value={recipe.cropSeconds ?? cropDraft}
              onChange={event => {
                const v = event.target.valueAsNumber;
                if (Number.isFinite(v)) setCropDraft(v);
                setEdits(previous => ({ ...previous, cropSeconds: v }));
              }} />
          </PreviewField>
        </div>
      )}
    </details>
    <TrainingPreviewSettings title={t('trainingStudio.yue2.method.previewTitle', 'Checkpoint previews (optional)')}
      enabled={cadence > 0} disabled={locked} onEnabledChange={enabled => setPreviewsOn(enabled)}
      enableLabel={t('trainingStudio.heartmula.enablePreviews', 'Render a full-song preview at every checkpoint')}
      enableInfo={t('trainingStudio.heartmula.enablePreviewsInfo', 'The server pauses training, renders one full song from the dataset, then resumes. The base model is rendered first and the final adapter last. Keep previews enabled to hear how the adapter changes.')}>
      <PreviewField label={t('trainingStudio.heartmula.previewSeconds', 'Preview maximum seconds')}
        info={t('trainingStudio.heartmula.previewSecondsInfo', 'The longest a preview may run, from 5 to 240 seconds. A full song ends on its own before that. Longer previews pause training for longer at every checkpoint.')}
        meta={t('trainingStudio.heartmula.previewSecondsMeta', 'default 240')}
        onReset={edited('seconds') ? resetPreview('seconds') : undefined}>
        <input className={INPUT} type="number" min={5} max={240} step={1} value={Number.isFinite(previewSettings.seconds) ? previewSettings.seconds : ''}
          disabled={locked} onChange={event => setPreview({ seconds: event.target.valueAsNumber })} />
      </PreviewField>
      <PreviewField label={t('trainingStudio.heartmula.previewSeed', 'Preview seed')}
        info={t('trainingStudio.heartmula.previewSeedInfo', 'The random seed used for every preview render, so previews across checkpoints are directly comparable rather than each landing on a different random take.')}
        meta={t('trainingStudio.heartmula.previewSeedMeta', 'default 42')}
        onReset={edited('seed') ? resetPreview('seed') : undefined}>
        <input className={INPUT} type="number" min={0} step={1} value={Number.isFinite(previewSettings.seed) ? previewSettings.seed : ''}
          disabled={locked} onChange={event => setPreview({ seed: event.target.valueAsNumber })} />
      </PreviewField>
      <PreviewField wide label={t('trainingStudio.heartmula.previewLyrics', 'Preview lyrics')}
        info={t('trainingStudio.heartmula.previewLyricsInfo', "Lyric Studio generation: the newest lyrics generated for this dataset's artist in Lyric Studio, so each checkpoint sings a song it never trained on. Falls back to the dataset's lyrics when there are none. Dataset lyrics: the first sung track's own lyrics. Either way the tags come from the dataset, because that is what training saw, and every checkpoint of a run uses the same words.")}
        onReset={edited('lyricsSource') ? resetPreview('lyricsSource') : undefined}>
        <StyledSelect accent="amber" className="w-full" disabled={locked} value={previewSettings.lyricsSource ?? 'generated'}
          onChange={value => setPreview({ lyricsSource: value === 'dataset' ? 'dataset' : undefined })}
          options={[
            { value: 'generated', label: t('trainingStudio.heartmula.previewLyricsGenerated', 'Lyric Studio generation (falls back to dataset lyrics)') },
            { value: 'dataset', label: t('trainingStudio.heartmula.previewLyricsDataset', 'Dataset lyrics') },
          ]} />
      </PreviewField>
      <p className="text-[11px] text-zinc-500 md:col-span-2">{t('trainingStudio.heartmula.previewSongHint', 'The tags come from the first sung track in this dataset. Tags and lyrics overrides below replace either choice.')}</p>
      <PreviewField wide label={t('trainingStudio.heartmula.previewTags', 'Tags override (optional)')}
        info={t('trainingStudio.heartmula.previewTagsInfo', "Replaces the preview track's own tags for the rendered sample. Leave blank to use the track's tags as-is.")}
        onReset={edited('tags') ? resetPreview('tags') : undefined}>
        <textarea className={`${INPUT} min-h-16 resize-y`} value={previewSettings.tags} disabled={locked} onChange={event => setPreview({ tags: event.target.value })} />
      </PreviewField>
      <PreviewField wide label={t('trainingStudio.heartmula.previewLyricsOverride', 'Lyrics override (optional)')}
        info={t('trainingStudio.heartmula.previewLyricsOverrideInfo', "Replaces the preview track's own lyrics for the rendered sample. Leave blank to use the track's lyrics as-is.")}
        onReset={edited('lyrics') ? resetPreview('lyrics') : undefined}>
        <textarea className={`${INPUT} min-h-20 resize-y`} value={previewSettings.lyrics} disabled={locked} onChange={event => setPreview({ lyrics: event.target.value })} />
      </PreviewField>
    </TrainingPreviewSettings>
    <p className="text-[11px] text-zinc-500 mt-2">{resume
      ? t('trainingStudio.heartmula.resumeSummary', 'Resume {{run}} with its saved recipe and preview settings.', { run: selectedResume })
      : t('trainingStudio.heartmula.recipeSummary', '3B backbone LoRA · rank {{rank}} · alpha {{alpha}} · {{steps}} steps', { ...recipe })}</p>
    <p className="text-[11px] text-zinc-500 mt-2">{t('trainingStudio.heartmula.adapterFolder', 'Adapters are saved in your global adapters folder under heartmula-lm-adapters/.')}</p>
    {!ready && <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">{t('trainingStudio.heartmula.prepareRequired', 'Prepare a non-empty dataset with no stale samples before training.')}</p>}
    {blockedReason && <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">{blockedReason}</p>}
    <div className="mt-4 flex items-center gap-3 flex-wrap">
      <button type="button" className={BUTTON} disabled={busy || (!batchIds && !ready) || !!blockedReason || (resume ? !resumable || !!batchIds : !valid)}
        onClick={batchIds ? chain : start}>
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
        {batchIds ? t('trainingStudio.heartmula.startBatch', 'Start batch') : resume ? t('trainingStudio.heartmula.resume', 'Resume run') : t('trainingStudio.heartmula.start', 'Start training')}
      </button>
      {busy && run && trainStarted && run.state === 'running' && <button type="button" className="text-xs text-amber-600 dark:text-amber-400 hover:underline" onClick={() => onCheckpoint(run.run)}>
        {t('trainingStudio.heartmula.checkpoint', 'Save and pause')}
      </button>}
      {busy && <button type="button" className="text-xs text-red-600 dark:text-red-400 hover:underline" onClick={onCancel}>{t('trainingStudio.heartmula.cancel', 'Cancel')}</button>}
      {run && <span className="text-[11px] text-zinc-600 dark:text-zinc-400">{run.state} · {run.step}/{run.totalSteps}</span>}
    </div>
    {showRun && <div className="mt-3" aria-live="polite">
      {run.error && <p role="alert" className="text-xs text-red-500">{run.error}</p>}
      {run.lossHistory.length > 0 && <TrainingChart epochs={[]} steps={run.lossHistory.map(point => ({ ...point, ep: point.step }))} target={0}
        config={{ series: ['loss'], defaults: ['loss'], xMax: run.totalSteps }} />}
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] tabular-nums text-zinc-500">
        <span>{t('trainingStudio.heartmula.loss', 'Loss')}: {run.loss?.toFixed(4) ?? '—'}</span>
        <span>{t('trainingStudio.heartmula.msPerStep', 'ms/step')}: {run.msPerStep?.toFixed(1) ?? '—'}</span>
        <span>{t('trainingStudio.heartmula.vram', 'Peak VRAM, MiB')}: {run.peakVramMiB?.toFixed(0) ?? '—'}</span>
        {eta && <span title={eta.previewsLeft
          ? t('trainingStudio.heartmula.etaPreviewsInfo', 'Includes {{n}} preview(s) at {{each}} each, measured from this run.', { n: eta.previewsLeft, each: eta.previewMs ? formatDuration(eta.previewMs) : '?' })
          : undefined}>
          {t('trainingStudio.heartmula.eta', 'ETA')}: {formatDuration(eta.ms)}
          {eta.previewsLeft > 0 && ` (${t('trainingStudio.heartmula.etaPreviews', 'incl. {{n}} preview(s)', { n: eta.previewsLeft })})`}
          {' · '}{t('trainingStudio.heartmula.etaDone', 'done ≈ {{time}}', { time: new Date(Date.now() + eta.ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })}
        </span>}
      </div>
    </div>}
    {ownLog.length > 0 && <details className="mt-2">
      <summary className="cursor-pointer text-[11px] text-zinc-600 dark:text-zinc-400">{t('trainingStudio.yue2.method.showLogs', 'Show training log (last 100 lines)')}</summary>
      <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-zinc-950 p-2 text-[10px] leading-4 text-zinc-300 whitespace-pre-wrap">{ownLog.slice(-100).map(line => line.message).join('\n')}</pre>
    </details>}
    </TrainingRunPanel>
  </div>;
}
