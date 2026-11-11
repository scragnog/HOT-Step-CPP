import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { gpusOptions, healthOptions, reportFailedJobOptions, reportLogsOptions } from '../../queries/shell-queue/settings';
import { useBackendStore } from '../../stores/backendStore';
import { backendModelsOptions } from '../../queries/global-models/options';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { ParamLabel } from '../shared/ParamLabel';
import { buildReport, modelLines, redact, renderBody, type LogSection } from './reportIssue';

const osName = () => {
  const ua = navigator.userAgent;
  return /Windows/.test(ua) ? 'Windows' : /Mac/.test(ua) ? 'macOS' : /Linux/.test(ua) ? 'Linux' : 'unknown';
};
const SHORT_TAIL = 60;

export function ReportIssuePanel() {
  const { t } = useTranslation();
  const health = useQuery(healthOptions()).data;
  const gpus = useQuery(gpusOptions()).data?.gpus;
  const logs = useQuery(reportLogsOptions()).data;
  const failed = useQuery(reportFailedJobOptions()).data;
  const backendId = useBackendStore(s => s.activeBackendId);
  const { ditModel, lmModel, vaeModel } = useGlobalParamsStore(s => s) as Record<string, unknown>;
  const catalogue = useQuery({ ...backendModelsOptions(backendId), enabled: !!backendId && backendId !== 'ace' }).data;
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [copied, setCopied] = useState(false);
  const unknown = t('settings.report.unknown', 'unknown');
  const b = health?.build;
  const system = [
    ['App version', health?.version || unknown],
    ['Build tag', b?.tag || unknown],
    ['Build sha', b?.sha || unknown],
    ['Build variant', b?.variant || unknown],
    ['Update channel', b?.channel || unknown],
    ['OS', osName()],
    ['GPU', gpus?.length ? gpus.map(g => `${g.name} (${Math.round(g.memoryMB / 1024)} GB)`).join(', ') : unknown],
    ['Compute backend', [health?.engine.backend, health?.engine.device].filter(Boolean).join(' / ') || unknown],
    ['Engine', backendId || unknown],
    ...modelLines(backendId, { ditModel, lmModel, vaeModel }, catalogue, unknown),
  ].map(([k, v]) => `${k}: ${v}`).join('\n');

  // All log text is redacted here, once, before it can reach the preview, the link or the clipboard.
  const sections = useMemo(() => {
    const names = { home: logs?.home, user: logs?.user };
    const cut = (lines: string[] | undefined, n: number) => (lines ?? []).slice(-n).map(l => redact(l, names));
    const failedLines = failed
      ? redact([`engine: ${failed.engine}`, `job: ${failed.jobId ?? 'none'}`, `attempt: ${failed.attempt}`, `error: ${failed.error ?? 'none'}`].join('\n'), names).split('\n')
      : [];
    const make = (n: number): LogSection[] => [
      { name: t('settings.report.lastFailed', 'Last failed job'), lines: failedLines },
      { name: 'Server log', lines: cut(logs?.server, n) },
      { name: 'Engine log', lines: cut(logs?.engine, n) },
    ];
    return { short: make(SHORT_TAIL), full: make(Infinity) };
  }, [logs, failed, t]);

  const parts = { title: title.trim() || 'Issue report from the app', description, system, sections: sections.short };
  const report = useMemo(() => buildReport(parts), [title, description, system, sections]); // eslint-disable-line react-hooks/exhaustive-deps
  const send = () => window.open(report.url, '_blank', 'noopener,noreferrer');
  const copyFull = async () => {
    await navigator.clipboard.writeText(`${parts.title}\n\n${renderBody({ ...parts, sections: sections.full })}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return <section className="settings-section">
    <div className="settings-section-header"><span className="settings-section-title">{t('settings.report.title', 'Report an issue')}</span></div>
    <div className="space-y-3 text-sm text-zinc-700 dark:text-zinc-300">
      <p>{t('settings.report.intro')}</p>
      <ParamLabel label={t('settings.report.titleLabel', 'Title')} info={t('settings.report.titleInfo')} />
      <input className="w-full rounded border border-zinc-500/30 bg-transparent p-2" value={title} maxLength={200} onChange={e => setTitle(e.target.value)} />
      <ParamLabel label={t('settings.report.descriptionLabel', 'What happened')} info={t('settings.report.descriptionInfo')} />
      <textarea className="h-32 w-full rounded border border-zinc-500/30 bg-transparent p-2" value={description} onChange={e => setDescription(e.target.value)} />
      <ParamLabel label={t('settings.report.previewLabel', 'Exact text that will be sent')} info={t('settings.report.previewInfo')} />
      <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-all rounded border border-zinc-500/30 p-3 text-xs">{`${report.title}\n\n${report.body}`}</pre>
      {report.trimmed && <p role="status">{t('settings.report.trimmed')}</p>}
      <div className="settings-actions justify-start">
        <button className="settings-save-btn" onClick={send}>{t('settings.report.send', 'Send to GitHub')}</button>
        <button className="settings-save-btn" onClick={copyFull}>{copied ? t('settings.report.copiedLogs', 'Logs copied') : t('settings.report.copyLogs', 'Copy full logs')}</button>
      </div>
    </div>
  </section>;
}
