# PERF.md — YuE2 Joint Training Performance (HOT-Step-CPP, M1 Max 64 GB)

Stand: 2026-10-02. Repo: `~/AI-Audio/HOT-Step-CPP-src`, Basis ggml `c044c6f0`. Patches in `engine/patches/` (sortiert angewendet). Ausführliche Details: `docs/perf-notes/fa-train-handoff-2026-09-30.md`, `engine/patches/README.md`.

## 1. Baseline und aktueller Stand

Messbefehl (Step 1, Gate-Metriken in der `"stage":"joint"`-Zeile):

Modell/Dataset:

```
cd ~/AI-Audio/HOT-Step-CPP-src
CKPT=~/AI-Audio/HOT-Step-CPP-Models/models/yue2/yue2_3b_int8_convrot.safetensors
DS=~/AI-Audio/HOT-Step-CPP-src/server/server/data/training/datasets/dataset_yotto/yue2-latents/aitk-prepared-7abafaf1-c8f4-4a46-9ae8-bcca11f33666/dataset.json
```

Benchmark-Lauf (1 Step):

```
/usr/bin/time -l engine/build/ace-train yue2-joint-train --checkpoint "$CKPT" --dataset "$DS" \
  --output /tmp/test-runNN --steps 1 --seed 42 --device MTL0 --cursor-weight 0 \
  --optimizer adamw-lm --ar-loss-weight 0.25 --kl-weight 0 --nar-crop-frames 0 \
  --weight-decay 0.1 --lr 0.0001 2>&1 | tee runNN.log | grep '"stage":"joint"'
```

Längerer Testlauf (5 Steps, lr 0.0002, `--device Metal`, mit Env-Flags; ältere Form des Befehls, Ergebnis lag bei Erstellung noch nicht vor):

```
GGML_METAL_OUT_PROD_TILED=0 GGML_METAL_FA_TRAIN_MM=1 engine/build/ace-train yue2-joint-train \
  --checkpoint "$CKPT" --dataset "$DS" --output /tmp/test-run \
  --steps 5 --seed 42 --device Metal --cursor-weight 0 --optimizer adamw-lm \
  --ar-loss-weight 0.25 --kl-weight 0 --nar-crop-frames 0 --weight-decay 0.1 --lr 0.0002
```

| Stand | Step 1 [s] |
|---|---|
| Baseline | 1533.1 |
| Aktuell (19ae548b) | 273.8 (−82,1 %) |

Hochrechnung bei 273.8 s/Step (ohne Aufschlag für längere Songs): 350 Steps ≈ 26,6 h, 400 ≈ 30,4 h, 750 ≈ 57 h. +10–15 % für längere Songs als der von Step 1.

Gate-Referenz (aktuell, run34/36/38 identisch):
- ar_ce 2.5117164487464816
- nar_mse 1.2940868182469722
- gradient_norm 0.14900318884557395

Ältere Referenzen:
- Single-pass, 16er-Tile: 2.511462543774956 / 1.2935168822972978 / 0.14860819428799363
- Two-pass (`GGML_METAL_FA_TRAIN_MM3_1P=0`): 2.512204314492374 / 1.2941176380249138 / 0.14923040272104537

Tests: `engine/build/fattn-train-test` (36/36 PASS), `--saved-check`, `--bench-yue2 --yue2-scale 0.25`, `yue2-mul-mat-k32-metal-test`, `yue2-acc-metal-test`.

## 2. Was funktioniert hat

| Commit | Maßnahme | Effekt (Step 1) |
|---|---|---|
| e950c0c9 | Weg 1: Single-pass Flash-Forward (Online-Softmax, Diagonalmatrix-MMA-Korrektur) | → 437.2 s; Gate-Werte verschoben (Rundung), neue Referenz freigegeben |
| 53afb185 | Weg 2: Backward-Occupancy-Tiles (Tiles halbiert auf ca. 11–13 KB Threadgroup-Memory) und Forward-Tile 8 nur für causal | → 345.9 s; Gate-Werte erneut verschoben, freigegeben |
| d4576bd1 | `kernel_mul_mm_k32_f32` mit K-Hälften (KLDS 20, 10240 B) | in der Kette bis 313.7 s enthalten, kein einzelner Messwert notiert |
| dd298772 | `kernel_cpy_f32_f32_v4` (vec4-Copy), auch für ACC/SET non-inplace | → 313.7 s; Gates bit-identisch |
| 19ae548b | Backward-NSG-Sweep: Defaults dQ 4 / dV 12 / dK 6 | → 273.8 s; Gates bit-identisch |

Mechanismus: Occupancy war durch Threadgroup-Memory begrenzt (Kernel mit 21–25 KB liefen mit 1 TG/Core). Halbierte Tiles brachten den größten Gewinn.

## 3. Ausgeschlossen

- ACC-Allocator-Aliasing (d5629dad, nur Handoff): −0,4 %, kein Nutzen, aus `.w7`-Backups zurückgenommen.
- Register als Limiter: `th_max` blieb bei 384–448 unverändert, also nicht das Problem.
- Forward-Tile 8 für NAR (non-causal): schlechter, deshalb nur causal.
- Instruments Shader Timeline: lieferte Zeiten, aber keine Limiter-Werte; durch NSG-Sweep ersetzt. Performance-Limiter-Daten stehen noch aus.
- Hypothese "Gate/Up-Gradientenassemblierung dominiert Copies": nur teilweise richtig. CPY_INFO zeigte auch Cont-of-Views, vec4-Copy brachte den Gewinn.
- convrot8-Backward via fp32-simdgroup-MMA (run41): 349.3 s statt 266.8 s (AR CE/KL bwd 11.8 -> 22.7 s, AR bwd 94.3 -> 139.2 s, NAR bwd 85.0 -> 111.8 s), Gates bit-identisch (bf16-Cast); fp32-MMA bringt auf M1 keinen Vorteil gegenüber skalarem 8x4-Kernel. Zurückgesetzt, nicht committet.
- convrot8_back_mm mit BN 16 (8 KB Threadgroup-Memory, gleiche Summationsreihenfolge, run43): 274.0 s statt 266.8 s, Gates identisch, Backward-Stufen unverändert im Rauschen. Kein Occupancy-Limiter für diesen Kernel. Zurückgesetzt.
- Instruments-Leerlauf (ca. 35 % GPU idle im 15-s-Fenster): Messartefakt, powermetrics zeigt 100 % GPU-Aktivität bei 1296 MHz im AR-Backward. Keine Host-/Sync-Lücken.
- Backward-NSG: Der Sweep ist abgeschlossen; Änderung von NSG ändert nur die Zeilengruppierung, Ergebnisse bit-identisch.

## 4. Offen und nächster Plan

Forward-Sweep gemessen (sweep2.log, `--bench-yue2 --yue2-scale 0.25`, ms, Rauschen ca. ±10 %): NSG6/NC16 am besten, NAR 60,4 / AR 30,0 (alt NSG8: NAR 71,8 / AR 36,6–36,8). Zweitplatziert NSG12/NC16 (61,1 / 31,8) und NSG4/NC16 (63,6 / 30,5). NC8 bei NSG6 schlechter (NAR 78,7 / AR 37,1). Alle 36/36 PASS.
Default gesetzt: NSG 6, NC 16 für causal und non-causal (ops.cpp, Backup `$HOME/bak/ggml-metal-ops.cpp.w10`). **Offen:** Build, Step-1-Messung (run39), Gate-Vergleich (causal NC 8→16 ändert Rundung, ggf. neue Referenz mit T69 freigeben), dann Patch/README/Commit.

Urspruenglicher Stand (Code editiert): Forward-1p-NSG/NC-Sweep.
- Kernel: `kernel_flash_attn_train_mm3_1p_n{4,6,8,12}_c{8,16}[_causal]_f32_d{64,128}` (32 Instanziierungen).
- Env: `GGML_METAL_FA_TRAIN_FWD_NSG` (4|6|8|12, Default 8), `GGML_METAL_FA_TRAIN_FWD_NC` (8|16, Default causal 8 / non-causal 16).
- Referenz: NAR ≈ 62.6 ms, AR ≈ 32.8 ms (NSG 8).
- Sweep-Befehl:

```
for n in 8 4 6 12; do for c in 8 16; do echo "== FWD_NSG=$n FWD_NC=$c"; \
env GGML_METAL_FA_TRAIN_FWD_NSG=$n GGML_METAL_FA_TRAIN_FWD_NC=$c engine/build/fattn-train-test 2>&1 | tail -1; \
env GGML_METAL_FA_TRAIN_FWD_NSG=$n GGML_METAL_FA_TRAIN_FWD_NC=$c engine/build/fattn-train-test --bench-yue2 --yue2-scale 0.25 2>&1 | grep -E "^(nar|ar) "; \
done; done 2>&1 | tee sweep2.log
```

- Danach: Defaults pro causal/non-causal wählen, Patch (Name muss nach `…zzzzzzzzzzzzzzzzzzzzzz-flash-attn-train-bwd-nsg.patch` sortieren, also ein `z` mehr), README, Handoff, Commit mit Messwert. NC ändert die Softmax-Rundung, daher mögliche Gate-Verschiebung: erst mit T69 besprechen.

Weitere Hebel:
- dQ/dV umstrukturieren, erst nach Performance-Limiter-Daten.
- `ggml_cont` der Gate/Up-Hälften; Saved-Slot-Copy mit v4-Kernel.
- Reale T/Tk eines Produktions-Jobs messen.
- f16 K/V für NAR (mit Hörtest).
- ACE-Step/MM3-Smoketest.
- Multi-Step-Vergleich mit `GGML_METAL_MM_K32=0`.
- Pro-Kernel-Limiter (ALU/Memory/Occupancy) für convrot8_back_mm, dK, dQ einzeln aus Instruments. Fensterdaten (AR-Backward, 15 s, 9.77 s GPU): convrot8_back_mm 23 %, dK 20 %, dQ 18 %, convrot8_mm fwd 13 %, dV 12 %. Counter-Mittel: ALU-Limiter 57 %, ALU-Util 48 %, Occupancy 25 %, Speicher/TG-Bandbreite niedrig.
- Speichercheck (AR-Länge 18478), `reuse_prefix` in Produktionskonfiguration prüfen.

Upstream-Merge (2026-10-04): 95 Upstream-Commits übernommen (Server/UI/YuE2 Covers, `--eval-base-loss`), unser `metal-acc-set-copy-width.patch` entfällt (identisch zu Upstream `metal-acc-set-cpy.patch`, #206). Gates und Zeit unverändert (run45).

## 5. Fallstricke

- `git -C engine/ggml checkout -- .` stellt den Patch-Stand NICHT wieder her (Submodul-HEAD 50723903 und Index sind älter als der gepatchte Baum, ggml.h/ggml.c/ops.cpp gehen verloren). Wiederherstellung per Replay auf `c044c6f0` plus alle Patches sortiert, Kontrolle mit `diff -rq` (muss leer sein).
- convrot8-Backward-Formen: nur der LM-Head (in 2048, out 184704, rows 128/45) hat wenige Threadgroups (64/32); die Transformer-Matmuls haben tausende.

Metal:
- Threadgroup-Memory ist der Occupancy-Limiter (ca. 32 KB/Core angenommen); Tile-Größe zuerst dort prüfen.
- `simdgroup_matrix` 8×8, `threadgroup_barrier` nötig.
- Gechunktes Q-Staging: `NPART = (NSG*8*(D/2) <= 2*NC*LD) ? 2 : ((NSG*8*(D/4) <= 2*NC*LD) ? 4 : 8)`; Spalten pro Lane `CPL = NC/4`. Im kv-Kernel NC→NQ.
- Es gilt `nsg*32 <= th_max` (Assert in ops.cpp).
- dK-Backward hat 2×Ssh Scratch (eigenes smem).

Build und Patches:
- NSG/NC sind Template-Parameter, per Host-Env zur Laufzeit gewählt (Host-Namen `..._n{N}[_c{NC}][_causal]_f32_d{D}`).
- Patch-Replay: `git -C engine/ggml archive c044c6f0 | tar -x -C $R/engine/ggml`, `git init` im `$R`-Root, alle Patches sortiert anwenden, committen, geänderte Dateien aus dem Repo kopieren, `git diff` ergibt den neuen Patch.
- Env-Flags: `GGML_METAL_FA_TRAIN_MM3_1P=0`, `GGML_METAL_CPY_V4=0`, `GGML_METAL_CPY_INFO=1`, `GGML_METAL_FA_TRAIN_INFO=1`, `GGML_METAL_FA_TRAIN_BDQ/BDV/BDK_NSG`, `GGML_METAL_MM_K32=0`, `GGML_METAL_FA_TRAIN_MM3[_BWD|_BWD_KV]=0`.
- Backups vor Edits: `$HOME/bak/*.w2 … *.w9` (w9 = vor Forward-NSG/NC-Edit).
- Commit-Trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` und `Claude-Session: https://claude.ai/code/session_019vqoJw7ciMCj9XTXbw68fe`.
- Stale Git-Locks (`HEAD.lock`, `index.lock`, `objects/maintenance.lock`, `tmp_obj_*`) nach Commits möglich; Löschen braucht Delete-Permission.
- device_bash/MCP-Verbindung bricht wiederholt ab; Tools neu laden.

Messfehlerquellen:
- Kein Hintergrund-I/O während Messung (z. B. USB-Kopie).
- Gate-Werte nur vergleichen, wenn Summationsreihenfolge gleich ist. NSG/Tile-Regrouping im Backward ist bit-identisch, NC/Softmax-Änderungen im Forward nicht.
- `yue2-convrot8-metal-tiled-test` (CPU-Referenz, 400 % CPU, sehr lang) nie parallel zu einem gemessenen Step laufen lassen.
- Metal-Kernelnamen stehen nicht in `ace-train`, sondern in `libggml-metal.*.dylib`.
- Bench `--yue2-scale 0.25` ist nur ein Proxy; maßgeblich bleibt Step 1 komplett.
- Kürzerer Song als Produktion: Aufschlag 10–15 % bei Hochrechnungen.
- Instruments-Datenmodellierung über 60 s ist zu langsam; 10–15 s Trace reichen.

## 6. Stand 2026-10-04: Recompute, Zweitmeinung, 5-Step-Lauf

Recompute (ausgeschlossen):
- Backward = Block-Level-Recompute (`yue2-aitk-block-executor.h` backward(): block() neu aufgebaut, nur Attention O+LSE gespeichert). convrot8_mm im AR-Backward-Fenster (1.31 s, 13 %) ist dieser Recompute.
- Speichern statt Recompute, AR S=18478, 28 Layer: alle Matmul-Ausgaben ~41 GB f32 / ~21 GB bf16; nur gate+up ~25 / ~12.4 GB (~50 % der Recompute-FLOPs); nur down-Eingang ~12.4 / ~6.2 GB (~25 %).
- Gewinn ca. 2-3 % am Step, Speicherrisiko bei 64 GB -> verworfen.

Zweitmeinung (Opus, nicht gemessen, nach Priorität):
- T3: convrot8-Forward gegen Upstream `kernel_mul_mm` (`test-backend-ops perf -o MUL_MAT`, q8_0, m=6144 n=15496 k=2048); Forward real ~2.9 TFLOPS. Nur weiter bei >=1.3x; bis -5..-7 %.
- T5: Thermik über 26 h prüfen (Netz, caffeinate, Step-Zeiten, powermetrics).
- T2: bf16-Rundung von w/dy einmal vorab statt pro Threadgroup, bit-identisch, ~2-3 %, ~+6 GB.
- T1: convrot8-Backward mit simdgroup_half8x8 (bf16-Produkte in fp32 exakt, w als half mit 2^G pro Matrix), -6..-9 %, 1-2 Tage, Gates wahrscheinlich, nicht sicher identisch.
- T4: dK/dV fusionieren ~1.5-3 %; T6: LM-Head-Chunk 128->512 (head-loss.h:36) <=1 %; T7: Restkernel fusionieren <=2-3 %.
- Generierung: G1 DPM++2M-NAR-Solver bauen/messen (bis ~-30 % Job, Hörtest); G2 QKV und gate/up zusammenfassen, ggml_swiglu_split, cont-Kopien weg (yue2-lm-graph.h:234-236, 318-327; Job -3..-6 %); G3 NAR-Gewichte F16 statt q8_0 (Job -3..-6 %, +1.3 GB); G4 prüfen, dass Produktion nicht den convrot-Checkpoint nutzt; G5 NAR-Attention mit Upstream flash_attn_ext messen.
- Urteil: vor dem 26-h-Training höchstens T3-Test; sonst starten.

5-Step-Lauf (run5, `--steps 5 --lr 0.0002`, ohne caffeinate):
- Step 1: 267.8 s, Gates bit-identisch zur Referenz. Alle Werte endlich, keine NaN. Peak-RSS 17.0 GB, Peak-Footprint 13.9 GB.
- Step-Zeiten: 267.8 / 298.9 / 180.3 / 1238.6 / 747.5 s. Step 4/5 ungültig: Mac ging in Idle/Maintenance Sleep (pmset-Log: Idle Sleep 12:49, Schlaf 900 s ab 12:51, danach DarkWake-Zyklen). Summe Schlaf im Lauffenster ~1300 s, passt zur Mehrzeit von ~1480 s.
- Swap 0, freier Speicher 89 %, AC, lowpowermode 0: Speicher/Batterie/Throttling nicht die Ursache.
- Songlängen streuen: 180-300 s pro Step sind Normalbereich, 266 s keine feste Zahl.
- Fallstrick: Training immer mit `caffeinate -dimsu` starten; Mac am Netz, Deckel offen oder externer Monitor; `pmset -g log | grep -E " Sleep | Wake | DarkWake "` zur Kontrolle.

### T3/T2-Messungen und Refresh-Befund (2026-10-04)
- T3 (convrot8-Forward vs. Upstream): `test-backend-ops perf -o MUL_MAT` (q8_0, m=4096 n=512 k=14336) = 6.10 TFLOPS; convrot8 fwd 3.35-4.24 TFLOPS (ar-gate-up 4.12, ar-o 3.35, ar-down 3.38, lm-head 4.24). Quantisierungsanteil ~8.5 ms (in=2048) / ~25 ms (in=6144); reiner Matmul ~4.3 TFLOPS -> Lücke ~1.4x. Struktur identisch zu kernel_mul_mm (64x32 Tile, 128 Threads, NK=32, HLDS=40 bankkonfliktfrei). Ursache offen, Obergrenze 2-5 % am Step (convrot8-Forward gesamt ~49 s, davon ~42 s Matmul). Der generische mul_mat q8_0 lag schon in B13/B14 bei 6.2-6.3 TFLOPS.
- T2 (Vorrunden von w/dy) verworfen: Probe-Kernel ohne alle Rundungen in den Lade-Pfaden (numerisch falsch, nur Timing): bwd -4..-7 % (ar-gate-up 465.6 -> 436.4 ms, ar-qkv -7.4 %, lm-head -4.6 %), fwd unverändert. Obergrenze <1 % am Step, dafür +5.6 GB und Umbau in Lader/Op-Signatur -> nicht lohnend.
- convrot8-Backward bleibt auch ohne Rundungen bei ~1.7-1.8 TFLOPS; B11-Ceilings: skalare fp32-FMA 3.28, half-MMA 6-7 TFLOPS. Einziger großer Hebel im Backward: half-MMA (T1), Muster wie B12 (kleiner TG-Speicher, half-Tiles).
- Backward convrot8 gesamt ~42 s/Step (16 %), Forward-Anteile (AR fwd, Refresh, Recompute, NAR) ~49 s (18 %).
- Refresh (18.8 s, ~7 % des Steps): `yue2-aitk-joint-step.h` überspringt ihn (`reuse_prefix`), wenn die NAR-Conditioning-Sequenz der AR-Sequenz entspricht (ganzer Song passt in kMaxFrames=24576, kein Crop, Planner nicht eingefroren); sonst B5 partial refresh (nur Zeilen nach dem gemeinsamen Prompt). Im 5-Step-Lauf entfiel der Refresh nur in Step 3. Weitere Senkung nur über das Rezept (kürzeres Lead-Sheet/`--abc-dropout`, Crop): ändert das Training, T69s Entscheidung. Diagnose pro Step: `YUE2_AITK_GRAD_DIAG=1` (reuse_prefix, ar_len, nar_cond_len), `YUE2_AITK_PROFILE=1` (B5 shared_rows).
- Historie (kleine Hebel, die sich summiert haben): B8+B9 zusammen -10.1 % (585 -> 526 s), B12 Occupancy (2.4 -> 4.2 TFLOPS). Wirkungslos/langsamer: E1 Q-Zeile in Registern (+2-4 %), NSG 24, Register-Prefetch, BN=16, fp32-MMA, ACC-Aliasing (-0.4 %).

## 7. Experimentelle Optionen (opt-in), Konzept 2026-10-04

Prinzip:
- Verworfene Hebel (Speicherbedarf oder veränderte Rechnung) werden später als experimentelle, abschaltbare Optionen aufgenommen (Flag oder Env-Variable). Standard bleibt der bit-identische Pfad mit den bisherigen Gate-Werten.
- Jede Variante bekommt eine eigene Gate-Referenz und wird getrennt dokumentiert.
- Für den PR an Scragnog: erst die bit-identischen Patches, experimentelle Optionen später und klar als experimentell markiert.

A) Speicher gegen Rechenzeit (Ergebnis bit-identisch, Gradienten unverändert):
- Aktivierungen speichern statt Recompute: AR S=18478, 28 Layer, alle Matmul-Ausgaben ~41 GB f32 / ~21 GB bf16; nur gate+up ~25 / ~12.4 GB; nur down-Eingang ~12.4 / ~6.2 GB. Gewinn ~2-3 % am Step.
- Vorgerundete bf16-Gewichte (T2): +5.6 GB. Probe-Kernel ohne alle Rundungen: bwd nur -4..-7 %, <1 % am Step.
- Speicherrahmen: Peak heute RSS 17.0 GB / Footprint 13.9 GB; Metal recommendedMaxWorkingSetSize ~55.7 GB; lange Songs (AR 18478) beachten.
- Mehr RAM allein löst den Engpass nicht: die Kernel sind Auslastungs-/Instruktionslimitiert, nicht Tabellen-limitiert.

B) Veränderte Rechnung (hier liegt der Gewinn):
- convrot8-Backward mit half-MMA (T1): geschätzt -7..-10 % am Step; ändert nur die Summationsreihenfolge (Rundungsrauschen), Gates nach bisherigen Erfahrungen (bf16-Endrundung) wahrscheinlich identisch, nicht garantiert. Risiko: Wertebereich in half (Über-/Unterlauf, abgeschnittene kleine dy-Werte) = systematischer Fehler.
- f16 K/V in der NAR-Attention (Generierung): +21-26 % Attention-Tempo, Klang minimal verändert, Hörtest nötig.

Einordnung der Gradientenwirkung:
- Rundungsrauschen (andere Summationsreihenfolge) ist unkritisch: das Rauschen durch wechselnde Trainingsbeispiele ist um Größenordnungen größer; frühere Rundungsänderungen verschoben ar_ce erst in der 4. Nachkommastelle.
- Systematische Verzerrung (Bereichsfehler, fehlende Terme) summiert sich über viele Steps und muss gemessen werden.

Validierung pro Variante:
1. A/B mit gleichem Seed über 5-20 Steps: Gradientennorm, Kosinus-Ähnlichkeit des Gradientenvektors zur Referenz (pro Schicht oder gesamt), Verlustkurve. Faustregel: Ähnlichkeit >0.999, Verlauf im Rauschen.
2. Kurzer Vergleichslauf auf kleinem Datensatz vor dem Einsatz in einem langen Lauf.
3. Eigene Gate-Referenz und Eintrag in PERF.md (Befehl, Messwerte, Datum).

### Frischer Klon (2026-10-05): Patch-Stack verifiziert
- Aufbau: `git archive HEAD` plus pristine ggml `c044c6f0` plus `engine/vendor`, dann `cmake -S engine -B engine/build` und Build von `fattn-train-test` und `ace-train` (Anleitung in `fa-train-handoff-2026-09-30.md`).
- Ergebnis: alle 35 Patches "applied patch" ohne Fehler, Build erfolgreich (nur Warnungen: ccache fehlt, 63 Compiler-Format-Warnungen), `fattn-train-test` 36/36 PASS. `diff -rq` des gepatchten ggml-Baums im Klon gegen den Arbeitsbaum (ohne .git/build): leer, also identisch.
- Offen: die 63 Compiler-Warnungen vor dem PR ansehen (z. B. `np * 4.0 / 1048576.0` Format-Hinweis); ACE-Step/MM3-Smoketest.

## 8. T1-Probe: half-simdgroup-MMA convrot8 Backward (Messung, noch nicht integriert)

Probe-Kernel hinter `GGML_METAL_CR8B_PROBE=1|2` (Default-Pfad unverändert, nur Bench, M1 Max):

- Probe 1 (ohne bf16-Casts, nicht exakt): bwd 3.9–4.5 TFLOPS (−58..−67 % bwd-Zeit).
- Probe 2 (exakte bf16-Rundung im Ladepfad, ohne Bereichsskalierung): 3.0–3.6 TFLOPS.
  ar-qkv 75.35 ms, ar-o 41.70, ar-gate-up 215.71 (vorher 465.9), ar-down 125.00,
  nar-qkv 46.31, nar-gate-up 132.27, nar-down 76.79, lm-head-chunk 31.99 (vorher 70.8).
- Standardkernel: 1.6–1.67 TFLOPS -> ca. 2x schneller. Schätzung Step: −25..27 s (~9–10 %) bei ~266 s.
- Offen für exakte Variante: half-Bereich (Skalierung 2^e je dy-Zeile, 2^G je Gewichtsmatrix, Unscale im Epilog),
  Realdaten-Bereichsprüfung, voller `yue2-convrot8-metal-tiled-test`, Gates, ein Step, A/B-Validierung.
  Optional: vorgerundete half-Gewichte (+5.6 GB), spart ~20–30 % der Kernelzeit (Cast-Kosten).

### 8.1 T1 exakte Variante (GGML_METAL_CR8B_PROBE=3), Messung

- Kernel `kernel_convrot8_back_mm_hs_f32` + Zeilen-Exponent-Vorlauf `kernel_convrot8_back_rowexp_f32`
  (dy-Zeile * 2^e, e aus absmax; Gewichte * 2^G, G pro Matrix aus den Scales, Fallback auf alten Kernel wenn half nicht exakt).
  Scratch: float2 pro Zeile (<1 MB). Kein nennenswerter Zusatzspeicher, Gradienten bitgleich im Test.
- Bench bwd: ar-qkv 75.8 ms, ar-o 41.9, ar-gate-up 217.4, ar-down 124.9, nar-gate-up 133.0, lm-head-chunk 32.5 (3.0-3.6 TFLOPS).
  Ein Vorlauf im Hauptkernel (je Kachel) kostete bei out=12288 +150 ms -> eigener Kernel noetig.
- `yue2-convrot8-metal-tiled-test` (voll, PROBE=3): ALL PASS, bwd bit-mismatch=0 in allen Faellen.
- Step 1 (joint, seed 42): 239.7 s statt 266-268 s (-26..28 s, ca. -10 %); AR transformer bwd 80.8 s, NAR flow bwd 76.4 s.
  Gates: ar_ce 2.511462543774956 und nar_mse 1.2935168822972978 identisch; gradient_norm 0.14860825662669813
  (Referenz 0.14860819428799363, rel. 4e-7, fp32-Summationsreihenfolge -> vereinzelte bf16-Rundungsflips).
- Offen: 5-Step-Verlauf, powermetrics, Standard-Schalter + Patch, neue Gate-Referenz fuer diesen Pfad (mit T69 abstimmen).

### 8.2 T1 abgeschlossen: Standard-Schalter, Kontrolle, Thermik

- Kernel jetzt Standard (Patch `zzzzzzzzzzzzzzzzzzzzzzzz-convrot8-back-half-mma.patch`), `GGML_METAL_CR8B=0` = alter Kernel.
  Probe-Kernel 1/2 entfernt.
- A/B 3 Steps, gleiche Einstellungen: Step 1 265.6 -> 239.7 s, Step 2 295.3 -> 267.6 s, Step 3 178.9 -> 158.6 s (-9.4..-11.3 %).
- 5-Step-Lauf (PROBE=3, caffeinate): 239.7 / 267.6 / 158.6 / 270.0 / 256.2 s. powermetrics: 112 aktive Samples,
  GPU 1295 MHz im Mittel (Min 1274), Thermal-Level durchgehend Nominal, 29-32 W, kein Trend (T5 bestanden).
- Gates Step 1: ar_ce, nar_mse identisch; gradient_norm 0.14860825662669813 (Referenz fuer diesen Pfad; alter Pfad 0.14860819428799363).
- Step 2 (Verlust vor dem zweiten Update, daher vergleichbar), Baseline / K32=0-Kontrolle / neuer Kernel:
  ar_ce 2.691772 / 2.691428 / 2.691108, nar_mse 1.456951 / 1.456899 / 1.456541, gradient_norm 0.179196 / 0.181481 / 0.178333.
  Die Abweichung des neuen Kernels (Loss ~2.5e-4 relativ, gradient_norm 0.5 %) liegt in der Groessenordnung der
  K32=0-Kontrolle (Loss 1.3e-4, gradient_norm 1.3 %): Adam-Update 1 verstaerkt kleinste Rundungsunterschiede (Vorzeichen bei Gewichten mit Gradient ~0).

## 9. Patch-Stapel: Wirkung auf das Ergebnis (Stand 2026-10-06)

Zusammengestellt aus `engine/patches/README.md` (keine Neumessung). "nicht dokumentiert" heißt: der README-Eintrag sagt dazu nichts.
Aktuelle Gate-Referenz (Step 1, Seed 42, Standardpfad): ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363
(bestätigt am 2026-10-06: 1-Step-Lauf nach dem LM-Head-Umbau, siehe 9.2).

### 9.1 Tabelle

| Patch | Wirkung auf das Ergebnis | Gate / Hinweis |
|---|---|---|
| bf16-out-prod, cpy-q-occupancy, quant-cpy-kquant, sched-unplaced-log, cudagraph-log, alloc-free-blocks | CUDA/Allokator/Diagnose; Rechnung selbst nicht verändert | nicht dokumentiert (plausibel ohne Wirkung) |
| f16-f32-accumulate | CUDA: F16-cuBLAS akkumuliert F32 (Korrektur, Ergebnis vorher falsch bei f16 + Adapter) | Hook 11 |
| mm-backward | nur mit `GGML_BACKWARD_MM=1`: andere Formulierung des Aktivierungsgradienten | opt-in, Gate-Angabe im README |
| zzzz-vulkan-train-ops | neue Vulkan-Ops | nicht dokumentiert |
| flash-attn-train | neue Ops + CPU-Referenz | Referenz für die Toleranztests |
| zzzz-yue2-convrot8-cpu | CPU-Kernel bitgleich zum Oracle | bit-exakt |
| metal-acc-set-cpy | Fehlerkorrektur (nicht-inplace ACC/SET) | nicht dokumentiert |
| zzzzz-metal-training-kernels | Metal-Seite der Trainingsops (CONVROT8, BF16_ROUND, SILU_BACK, REPEAT_BACK, RMS_NORM_BACK, IM2COL-Fix, OUT_PROD, FA-train skalar). Im README beschrieben (Sammeldiff vom 2026-09-30). `GGML_METAL_OUT_PROD_TILED` standardmäßig an: float32-Toleranz gegen den zeilenweisen/CPU-Kernel, NICHT bitgleich (`yue2-out-prod-metal-tiled-test`); Rest ohne Genauigkeitsangabe pro Op | OUT_PROD_TILED: ~31 % schneller pro Step im NAR-LoRA-Lauf; `=0` = ungekachelte Referenz |
| flash-attn-train-saved-forward | bitgleich | saved-check |
| flash-attn-train-mm3-forward/-dq/-dkdv | Toleranz, NICHT bitgleich zum skalaren Kernel (1e-4; dQ ~1.8e-6, dK/dV ~1.4e-6) | erst opt-in, durch `mm3-default-on` Standard |
| flash-attn-train-mm3-default-on | schaltet die B3-Kerne standardmäßig ein: Standardergebnis ist damit das der toleranzgeprüften mm3-Kerne; die skalaren Kerne (alle drei Variablen `=0`) bleiben die Bitgleichheits-Referenz | README nennt keinen Gate-Wert für diesen Schalter |
| flash-attn-train-causal-hint / -causal-variants / -scratch-pad | bitgleich | Gate unverändert |
| mul-mat-k32-f32 | Standard, NICHT bitgleich (geblockte Summation), Toleranz 1e-4 | `yue2-mul-mat-k32-metal-test` |
| mul-mat-k32-half-k | Umbau der Tiles; "same k order per accumulator as before" (Ergebnis nicht verändert erwartet, Bitgleichheit nicht ausdrücklich behauptet) | `yue2-mul-mat-k32-metal-test` ALL PASS |
| convrot8-fwd-half-tiles | bitgleich (0 Abweichungen) | tiled test ALL PASS |
| convrot8-back-vec-bm64 | bitgleich (Summation weiter aufsteigend) | tiled test ALL PASS |
| flash-attn-train-mm3-single-pass | Standard, NICHT bitgleich (Rundungsreihenfolge) | Toleranz, 36/36 PASS |
| flash-attn-train-occupancy-tiles | verschob die Gate-Referenz (kausale Kachel 8) | ar_ce 2.5117164487464816, nar_mse 1.2940868182469722, gradient_norm 0.14900318884557395 |
| metal-cpy-vec4 | bitgleich (reine Kopie) | Gate unverändert |
| flash-attn-train-bwd-nsg | bitgleich | Gate unverändert |
| flash-attn-train-fwd-nsg-nc | Kachel 8 -> 16 ändert die Rundung zurück | Gate zurück auf ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363 |
| metal-bin-threads, metal-im2col-ic (Metal) | Thread-Obergrenze bei Binäroperationen bzw. im2col parallel über Eingangskanäle (N == 1, 1D-Faltungen); Ergebnis nicht verändert erwartet | kein eigener README-Absatz |
| zzz-yue2-bf16-round | neue Op BF16_ROUND (F32 -> BF16 -> F32); definiert die bf16-Rundung | kein eigener README-Absatz |
| flash-attn-train-kv-grad-start | API: ab welcher KV-Zeile Gradienten fließen (op_params Slot 4); Ergebnis nur dort verändert, wo das Training es setzt | nur im Sammelabsatz erwähnt |
| convrot8-back-half-mma | Standardpfad bitgleich (Step 1: gradient_norm 0.14860819428799363); LM-Head standardmäßig auf dem alten Kernel, nur mit `GGML_METAL_CR8B_LMHEAD=1` im half-Kernel (3.6e-4 der Werte abweichend, 85 % um eine bf16-Stufe) | siehe 9.2 |

### 9.2 convrot8-back-half-mma: Prüfung auf echten Daten (`GGML_METAL_CR8B_CHECK=1`)

- Beide Kerne rechnen, Vergleich bitweise auf dx. Transformer-Matrizen: 0 Abweichungen (qkv/o/gate-up/down, je 56 Aufrufe, 1.4-4.3 Mrd. Werte).
- LM-Head (in 2048, out 184704): 11173 von 31.3 Mio. Werten abweichend (1 bf16-Stufe: 9512, mehr: 1661), größte Abweichung 3e-8 bei Maximalwert 2.3e-5.
  Ursache: Softmax-Gradienten einer Zeile überspannen mehr als der half-Exponentenbereich nach der Zeilenskalierung.
- Konsequenz: LM-Head standardmäßig auf dem alten Kernel (Verlust ~4-5 s/Step), `GGML_METAL_CR8B_LMHEAD=1` als experimentelle Option (voller Gewinn ~26 s/Step, gradient_norm 0.14860825662669813).
- Beobachtung: Der `K32=0`-Kontrolllauf wich in Step 2 trotz bitgleichem Step 1 ebenfalls vom Baseline ab; warum, ist nicht geklärt (Baseline-Reproduzierbarkeit zwischen zwei Läufen nicht geprüft).

- Bestätigung (2026-10-06): Standardpfad (LM-Head auf altem Kernel), 1 Step: step_ms 245.7 s (Baseline 265.6 s, -7.5 %),
  ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363 (bitgleich zur Referenz).
  Mit `GGML_METAL_CR8B_LMHEAD=1`: 239.7 s, gradient_norm 0.14860825662669813. Der Unterschied (~6 s/Step) ist der LM-Head-Backward.

### 9.3 Herkunft der 36 Patches (Git-Verlauf, Stand 2026-10-06)

Maßstab: erste Anlage der Datei (Autor) und ob sie im gemeinsamen Stand mit scragnogs Repo (Merge-Base 068564ba, 2026-10-02) schon vorhanden war.

- **15 Patches aus scragnogs Repo** (Autor Rob, ein Metal-Patch "Engineer (Opus 5.5)"), im Fork HOT-ggml seit 2026-10-05 als Commits: alloc-free-blocks, bf16-out-prod, cpy-q-occupancy, cudagraph-log, f16-f32-accumulate, flash-attn-train, metal-acc-set-cpy, metal-bin-threads, metal-im2col-ic, mm-backward, quant-cpy-kquant, sched-unplaced-log, yue2-convrot8 (CUDA), yue2-bf16-round, vulkan-train-ops.
- **21 Patches von uns** (Autor "T69" im Git, 2026-09-23 bis 2026-10-05), nicht im Fork:
  - 09-23: yue2-convrot8-cpu, metal-training-kernels (Sammeldiff, 4 Commits)
  - 09-30: flash-attn-train-kv-grad-start, -saved-forward, -mm3-forward, -mm3-dq, -mm3-dkdv, -mm3-default-on
  - 10-01: flash-attn-train-causal-hint, -causal-variants, -scratch-pad, -mm3-single-pass, -occupancy-tiles, mul-mat-k32-f32, mul-mat-k32-half-k, convrot8-fwd-half-tiles, convrot8-back-vec-bm64
  - 10-02: metal-cpy-vec4, flash-attn-train-bwd-nsg, flash-attn-train-fwd-nsg-nc
  - 10-05: convrot8-back-half-mma (2 Commits)
- Für die PR relevant: Der Fork enthält die Ops (FLASH_ATTN_TRAIN CPU+CUDA, CONVROT8 CUDA, BF16_ROUND), aber keine Metal-Kernel dafür und keinen CPU-CONVROT8-Kernel; das sind unsere 21.
- Der Metal-Shader liegt im Fork unter `src/ggml-metal/kernels/` (nicht mehr `ggml-metal.metal`), Basis ggml-org 353b63b4: alle 21 müssen übertragen werden.

## 10. Attention-Backward: Shader-Profil und negativer Versuch (Stand 2026-10-06)

Methode: GPU-Capture eines AR-causal-Graphen (`MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=8 fattn-train-test --bench-yue2 --yue2-only ar --yue2-scale 0.25 --bench-iters 3`), in Xcode geöffnet, Reiter Performance > Shaders. Die Hardware-Counter blieben N/A (Gerät als inkompatibel gemeldet); die Compiler-Statistik und die Laufzeitkosten pro Instruktionsklasse sind trotzdem verfügbar.

Anteile im Graphen: dK (`dk_mm_n6`) 37,1 %, dQ (`dq_mm_n4`) 28,5 %, Forward (`mm3_1p_n6_c16`) 18,7 %, dV (`dv_mm_n12`) 15,0 %; alles andere unter 1 %.

| Kernel | Temp-Register | Max. Occupancy | Spills | Integer | Float-Matrix | Wait | Memory |
|---|---|---|---|---|---|---|---|
| dK | 128 | 12,5 % | 96 B | 36,7 % | 8,2 % | n/a | n/a |
| dQ | 128 | 12,5 % | 80 B | 40,8 % | 9,6 % | 15,6 % | 14,6 % |
| dV | 108 | 14,6 % | 0 | 37,7 % | 11,7 % | n/a | 12,1 % (Sync 10,5 %, Control Flow 13,9 %) |

Befund: Integer-/Adress-Arbeit kostet je Kernel etwa das Vierfache der Matrix-Arbeit. dK hält K, V und den Akkumulator als Fragmente in Registern (3 x 16 Fragmente = 96 von 128 Registern).

Versuch A (bit-exakt, dK/dV, nur causal): analytische Block-Liveness statt Masken-Scan mit `simd_any` und Barriere, dazu schneller Pfad für vollständig sichtbare Innen-Tiles ohne Bound-/Masken-Tests. `fattn-train-test` 36/36 PASS. `--bench-yue2 --yue2-only ar --yue2-scale 0.25 --bench-iters 20`, Backward 115,43 -> 118,28 ms (Rauschen ca. +-10 %), kein Gewinn. Verworfen, nicht eingebaut.

Einordnung: Der Overhead entsteht vermutlich pro 8er-Query-Tile (Staging mit 64-Bit-Adressen, 4-5 Barrieren) und nicht in der Elementlogik. `NQ` 16 -> 8 (Patch occupancy-tiles) bleibt die bessere Wahl, weil Register und Threadgroup-Speicher die Occupancy begrenzen. Weitere Kandidaten, jeweils mit unsicherem Nutzen: Staging-Adressen je Thread vorberechnen (Base-Pointer pro g), Q/dO-Tile für mehrere Tiles im Voraus laden (Double-Buffering), Fragmente in half halten (ändert die Rechnung). Priorität niedrig; zuerst die Convrot8-Hebel (vorgerundete Half-Gewichte, LM-Kopf).

## 11. Vorgerundete Half-Gewichte (PREW) und Opus-Zweitmeinungen (Stand 2026-10-07)

PREW (`GGML_METAL_CR8B_PREW=1`, Opt-in): Die Gewichte des Convrot8-Backward werden einmal auf dem Host als `half(bf16(q*bf16(s)) * 2^wexp)` vorberechnet (+2 Byte pro Gewicht), der Kernel lädt sie direkt in den Threadgroup-Speicher. Ergebnis: bit-identisch, am Step kein messbarer Gewinn, **verworfen**.

Messung:

- Bit-Test `yue2-convrot8-metal-tiled-test` mit PREW: alle 9 Fälle `bit-mismatch=0` (fwd und bwd), ALL PASS.
- Mikrobenchmark `--bench` (Backward, ohne PREW -> mit PREW): ar-qkv 78,6 -> 65,6 ms, ar-o 43,2 -> 36,7, ar-gate-up 224,7 -> 199,9, ar-down 127,9 -> 109,6, nar-qkv 48,2 -> 40,4, nar-gate-up 138,8 -> 120,7, nar-down 78,7 -> 67,2 (-11 bis -17 % je Matmul; Forward und LM-Kopf unverändert).
- Step 1 (Seed 42, Standard-Parameter, hs-port-Binary): 246,3 s ohne, 248,5 s mit PREW. Gate-Werte unverändert (ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363). AR transformer backward 82,7 -> 82,3 s, NAR flow backward 76,5 -> 77,8 s, AR forward 26,0 -> 26,3 s, NAR forward 30,6 -> 31,1 s. Alle Differenzen im Rauschen (etwa +-1 bis 2 s). Zusatzspeicher 5,64 GB, Peak-Footprint des Prozesses 18,65 GB (kein Vergleichswert ohne PREW).

Einordnung:

- Der Mikrobenchmark (-13 % je Matmul) überträgt sich nicht auf den Step. Der Backward-Matmul-Anteil am Step ist kleiner als der Profilanteil von 23 % im AR-Backward nahelegt, oder wird im echten Graphen teilweise von anderen Kernels verdeckt. Erwartet waren 2-4 s (etwa 1 %); die frühere Schätzung von 7-10 % war falsch.
- Entscheidungsregel war: unter 3 s Gewinn -> Code entfernen. Er ist aus `hgport` und `hs-port` entfernt. Der Stand der Probe liegt als `prew-experiment.patch` in diesem Ordner.
- Zwei Fehler der ersten Fassung, falls jemand es erneut versucht: (1) `ggml_backend_buffer_is_host` ist bei allen Metal-Buffertypen `false`, die Aktivierungsbedingung muss auf "nicht `_Private`" lauten; (2) ein Cache-Schlüssel nur aus der Gewichtsadresse kollidiert, wenn ein freigegebener Buffer von einer anderen Matrix wiederverwendet wird (im Bench passiert). Der Patch enthält beide Korrekturen (Schlüssel: Adresse, Form, Stichproben-Hash).

Opus-Zweitmeinungen: siehe `opus-sessions-2026-10-06.md` (YuE2-Training und ACE-Step, jeweils im Wortlaut mit Einordnung).

Offen im Training nach diesem Ergebnis: A_NAR messen (NAR flow backward 77,8 s, etwa 31 % des Steps, nicht profiliert), dann Entscheidung über den Attention-Backward-Umbau (dK/dV-Fusion, A_AR etwa 40 s) und Cloud-Probelauf.

## 12. Attention-Bench in Vollgröße: A_AR, A_NAR und Anteil am Step (Stand 2026-10-07)

Messung `fattn-train-test --backend metal --bench-yue2 --yue2-only {ar,nar,nar-noskip} --bench-iters 5` (2 Warm-up, Vollgröße, Nh 16 / Nkv 8 / D 128, Arm "fused f32", pro Layer und Aufruf). Während der Läufe war möglicherweise leichte Hintergrundlast aktiv (Browser).

| Fall | Q | KV | Forward | Backward | Fwd-TFLOPS | Bwd-TFLOPS (nominal) |
|---|---|---|---|---|---|---|
| ar (causal) | 15496 | 15496 | 393,3 ms | 1750,0 ms | 2,50 | 1,41 |
| nar (mit Prefix-Skip B1) | 9502 | 24573 | 730,9 ms | 1940,2 ms | 2,62 | 2,47 (nominal, nicht um den Skip korrigiert) |
| nar-noskip | 9502 | 24573 | 732,2 ms | 3112,2 ms | 2,61 | 1,54 |

Definitionen laut Quelltext: "bwd ms" = (Forward+Backward-Graph) minus Forward-only-Graph, also die Backward-Kernel einschließlich ihrer internen S-Neuberechnung, **ohne** den Forward-Recompute des Block-Executors. Die Bwd-TFLOPS zählen 2,5 x die Forward-FLOPs und sind beim Fall "nar" nicht um den Prefix-Skip korrigiert (die Zahl 2,47 ist zu hoch; Skip: 1,94 gegen 3,11 s, also 1,6-fach schneller).

Korrektur einer ersten Lesart: Der causale AR-Backward ist **kein** Ausreißer. Real laufen beide Backward-Varianten bei etwa 1,4-1,5 TFLOPS (ar 1,41, nar-noskip 1,54), der Forward bei etwa 2,5-2,6. Der Backward ist durchgängig etwa 1,7-fach weniger effizient als der Forward.

Eichung gegen das Instruments-Profil (AR-Backward-Fenster: dK+dQ+dV = 50 % von 82,7 s = 41 s): Bench 28 x 1,75 = 49 s, also etwa 20 % höher; der Bench ist ein Obergrenzen-Schätzer.

Backward-Struktur (Code, `ggml-metal-ops.cpp` Dispatch und `hotstep_train.metal`, gezählt an den `simdgroup_multiply_accumulate`-Zeilen): drei getrennte Kernel, die jeweils S = QK^T neu rechnen: dQ (3 Matmuls: S, dP, dS*K), dV (2: S, P^T*dO), dK (3: S, dP, dS^T*Q). Zusammen **8 Matmul-Einheiten** statt der 5 des nominalen Aufwands (daher "2,5 x Forward"). Die reale Rate ist damit 1,41 x 8/5 = 2,26 TFLOPS (ar) bzw. 1,54 x 8/5 = 2,46 TFLOPS (nar-noskip), also **auf Forward-Niveau (2,5-2,6)**. Die Backward-Kernel sind nicht schlechter ausgelastet als der Forward; der Verlust ist algorithmisch (S dreimal, dP zweimal berechnet).

Korrektur einer ersten Fassung dieses Abschnitts: Der Forward-Recompute der Attention im Block-Backward entfällt bereits (B2, 2026-09-30 umgesetzt: O+LSE aus dem Forward werden als "saved" kopiert, bit-identisch, Tape +3,6 GB AR / +2,2 GB NAR). Der "bwd ms"-Wert des Benchmarks ist deshalb die gesamte Attention-Zeit der Backward-Stufen. (Hinweis: `saved_attention` in `yue2-aitk-graph.h` wird nur gesetzt, wenn der Executor einen Slot liefert; die Wirkung ist in den Step-Zeiten sichtbar, weil B2 laut Backlog gemessen -17 % brachte.)

Rekonstruktion des Steps (28 Layer im AR und im NAR, Nachweis `yue2-nar-graph.h`; Matmul-Zeiten aus dem Convrot8-Bench; Summe der Teile gegen die gemessene Stufenzeit):

| Stufe (gemessen) | Attention (Bench) | Matmul (Bench) | Summe der Teile |
|---|---|---|---|
| AR adapted forward 26,0 s | 11,0 s (28 x 0,393) | | |
| Refresh 18,9 s | etwa 11 s | | |
| AR transformer backward 82,3 s | 49 s (28 x 1,75) | etwa 25 s (Matmul-Recompute 11,7 + Bwd 13,3) | 74 s (-10 %) |
| NAR forward 31,1 s | 20,5 s (28 x 0,731) | | |
| NAR flow backward 77,8 s | 54 s (28 x 1,94) | etwa 15 s | 69 s (-11 %) |

Abgeleitet (Genauigkeit etwa +-15 %): Die Attention macht etwa **145 s von 248 s (59 %)** aus: etwa 42 s Forward (drei Aufrufe pro Layer und Step: AR adapted, Refresh, NAR) und etwa 103 s Backward. Die Summe der Teile liegt in beiden Backward-Stufen 10-11 % unter der gemessenen Zeit; der Rest sind Elementwise-, Norm- und Adapter-Operationen. Die Backward-Matmuls (PREW, Abschnitt 11) sind kein großer Hebel.

Kandidaten (alle ungemessen; Schätzungen in Sekunden pro Step bei 103 s Attention-Backward):

1. dK/dV fusionieren (S und dP einmal, dann beide Akkumulationen): 8 -> 7 Matmuls, bit-identisch möglich (gleiche Akkumulationsreihenfolge je Ausgabe), Risiko Register-Druck (dK hält schon 96 von 128 Registern, Spills wahrscheinlich). Erwartet etwa -5 bis -10 s.
2. dS materialisieren: Der dK/dV-Kernel schreibt dS (f32) kopfgruppenweise in einen Scratch-Puffer (je Kopf bis 480 MB causal AR, 934 MB NAR; Gruppen von 2-4 Köpfen), ein eigener dQ-Kernel liest dS und multipliziert mit K in derselben Reihenfolge wie heute. Matmuls: S, dP, dV, dK im fusionierten Kernel plus 1 für dQ = 5 statt 8; Speichertraffic nur etwa 2 % der Kernelzeit. Bit-Identität plausibel, aber nicht garantiert (MMA-Reihenfolge muss exakt nachgebaut werden). Aufwand mehrere Tage. Theoretisch bis -35 s, realistisch -20 s.
3. FA2-Stil (dQ per Float-Atomics aus dem dK/dV-Kernel): ähnlicher Gewinn wie 2, aber Summationsreihenfolge ändert sich, also nur als Opt-in mit eigener Gate-Referenz; Verfügbarkeit von Float-Atomics auf Apple7 ungeklärt. Nicht empfohlen, solange 2 offen ist.
4. Basis-Effizienz aller Attention-Kernel (Forward und Backward bei 2,3-2,5 von etwa 10,4 TFLOPS FP32-Peak; Xcode: Integer-Arbeit etwa das Vierfache der Matrix-Arbeit): allgemeine Verbesserung, Versuch A brachte nichts.
5. Threadgroup-Reihenfolge im causalen Fall: Die Hypothese "Tail-Imbalance" ist nach der Messung schwach (auch nicht-causal gleiche Rate); nur als Nebenversuch.

Lehre aus PREW: Erst ein Step-Lauf mit dem geänderten Kernel zählt.

### 12.1 Operandentausch-Test (2026-10-07)

`engine/tools/simdgroup-swap-test.mm` (standalone Metal): `simdgroup_multiply_accumulate` für S = Q·Kᵀ gegen Sᵀ = K·Qᵀ (D = 128, 8er-Blöcke, gleiche d-Reihenfolge, f32, Akku ab 0). Ergebnis auf M1 Max: **bitgleich**, 0 Abweichungen bei je ~1 Mio. Ausgaben in 5 Datenklassen (normal, weiter Exponent, Denormals, ±0, attention-ähnlich), Fast-Math an und aus. Folge: Variante 2a (KV-Kernel schreibt dS, leichter dQ-Kernel liest es) kann ohne Änderung der dQ-Numerik gebaut werden. Offen bleibt, ob dS = P·(dP − δ) im KV-Kernel dieselben Zwischenwerte (exp, dP) bitgleich liefert; das prüft `fattn-train-test` (36 Fälle) am fertigen Kernel.

### 12.2 dS-Materialisierung (DSW) im AR: bitgleich, Backward -22 %, Step -9,7 s (2026-10-07)

Opt-in `GGML_METAL_FA_TRAIN_DSW=1` im Fork-Port (hgport 0c751fa..d716f54): der dK-Kernel schreibt dS als 8x8-f32-Tiles in einen dauerhaften Scratch, ein leichter dQ-Kernel liest sie (kein S, dP, exp, keine Q/dO-Fragmente). Nur causal mit `kv_grad_start == 0` (AR), D 64/128; dV läuft vorab allein, dann pro Gruppe von KV-Köpfen (`GGML_METAL_FA_TRAIN_DSW_HKG`, Standard 2) dK, Barriere, dQ, Barriere.

- Bitgleichheit: `fattn-train-test --dsw-check` (S 8..1537, hkg 1/2/3, nc 8/16/32, nsg 4/8) all PASS, packed/dQ/dK/dV. Voraussetzung war, dass der Scratch-Wert wie im alten dQ-Kernel gerundet wird: `fma(scale, S, mask - LSE)` (`GGML_METAL_FA_TRAIN_DSW_VAR=1`, Standard). Mit der Rundung des dK-Kernels (VAR=0) und ohne fma (VAR=2) weicht dQ um 1 ulp ab: die beiden alten Kernel runden dS heute schon unterschiedlich.
- Bench AR (Vollgröße): Backward 1765 -> 1372 ms pro Layer (-22 %), Forward unverändert (393-395 ms). Knöpfe-Sweep flach (1346-1479 ms); hkg=4 nur 2 % schneller bei 7,7 GB Scratch, daher Standard hkg=2, nc=16, nsg=8.
- Step 1 (Seed 42, hs-port, DSW=1): 236,6 s statt 246,3 s. Gate bit-identisch (ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363). AR transformer backward 82,3 -> 72,7 s; NAR-Stufen unverändert. Scratch 3664 MiB, Peak 16,85 GB. Einzellauf; Vergleichslauf mit DSW=0 im selben Baum steht aus.
- Nicht abgedeckt: NAR mit Prefix-Skip (dQ muss dort für die Prefix-Schlüssel weiter den alten Kernel nutzen; erwarteter Gewinn nur etwa 5-6 s pro Step). Nur im Fork-Port, noch nicht im Patch-Stand von `HOT-Step-CPP-src/engine/ggml`.

### 12.3 Kernel-Anteile nach DSW und dK/dV-Fusion (negativ, 2026-10-07)

Anteile des AR-Backwards nach DSW (1359 ms pro Layer; Diagnose `GGML_METAL_FA_TRAIN_DSW_SKIP`, Ergebnis dabei absichtlich falsch): dK mit dS-Schreiben 810 ms (60 %), dV 402 ms (30 %), dQ-Leser 108 ms (8 %).

Versuch: dK und dV in einem Kernel (dV-Akkumulator im dK-Kernel, V als Threadgroup-Speicher statt Fragmenten, NSG 4 wegen 32 KB; Patch archiviert in `dsw-fuse-experiment.patch`, hgport 1e6d847, danach zurückgenommen). Bitgleich (dsw-check PASS mit beiden P-Rundungsvarianten), aber **kein Gewinn**: Bench AR Backward 1350 (getrennt) gegen 1344 ms (fusioniert). Der fusionierte Kernel wird also um etwa die Zeit des eingesparten dV-Kernels langsamer. Wahrscheinlichste Gründe (nicht per Xcode geprüft): NSG 4 statt 6, V-Fragmente aus Threadgroup-Speicher in jeder Kachel, Registerdruck mit zwei Akkumulatoren.

### 12.4 DSW im Produktions-Baum, A/B-Step (2026-10-07)

DSW (§12.2) auf den Patch-Stand von `HOT-Step-CPP-src/engine/ggml` übertragen (dort liegen die Kernel in `ggml-metal.metal`; der Patch lief mit reduziertem Kontext durch und hatte die Scratch-Zeile zunächst in den dQ-Kernel gesetzt, von Hand in den dK/dV-Kernel verschoben). `fattn-train-test --dsw-check`: all PASS. Step 1, Seed 42, `perf-logs/dsw-step.sh`:

| | DSW=0 | DSW=1 |
|---|---|---|
| step_ms | 246475 | 238223 (-8,3 s, -3,4 %) |
| AR transformer backward | 81,7 s | 73,8 s |
| NAR flow backward | 77,7 s | 77,3 s |
| peak memory footprint | 13,0 GB | 16,8 GB |

Gate bit-identisch (ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363). Der Scratch (3664 MiB bei hkg=2) erklärt den Peak-Anstieg. Der frühere Peak von 18,65 GB (PREW-Lauf) lag nicht an DSW. Opt-in bleibt `GGML_METAL_FA_TRAIN_DSW=1`; ob es Standard wird, ist offen. Im Produktions-Baum nicht committet (liegt ungestaged neben den übrigen Patches).

### 12.5 DSW als Standard und NAR-Hybrid, A/B im Produktions-Baum (2026-10-07)

DSW ist jetzt Standard (`GGML_METAL_FA_TRAIN_DSW=0` schaltet auf die Recompute-Kernel zurück; Scratch-Größe `..._DSW_HKG`, Standard 2). Neu: NAR-Hybrid für non-causal ohne Maske mit `kv_grad_start` (hgport c2ec6f7; im Produktions-Baum per `dsw-nar-port.py`): Der alte dQ-Kernel rechnet vorab die Prefix-Schlüssel `[0, jstart)` (`jstart` = `kv_grad_start` auf den dK-Block von 48 abgerundet, `jlim`-Argument), sein f32-Akkumulator liegt als dQ im Ausgang, der DSW-Leser setzt die Kette dort fort; der dK-Kernel schreibt dS nur für Schlüssel ab `jstart` (Scratch im NAR etwa 1,45 GB statt 3,66 GiB). `--dsw-check` hat jetzt NAR-Fälle (ohne Maske, auch kvgs 47/48/0): all PASS, Testklon und Produktions-Baum.

- Bench NAR (Testklon): Backward 1934 -> 1816 ms pro Layer (-6 %). Kleiner als der AR-Gewinn, weil 61 % der dQ-Arbeit (Prefix) beim alten Kernel bleiben.
- Step 1, Seed 42, Produktions-Baum (`perf-logs/dsw-step.sh`):

| | DSW=0 | DSW=1 |
|---|---|---|
| step_ms | 245033 | 232490 (-12,5 s, -5,1 %) |
| AR transformer backward | 81,7 s | 72,0 s |
| NAR flow backward | 76,2 s | 73,3 s |
| peak memory footprint | 13,0 GB | 16,8 GB |

Gate bit-identisch (ar_ce 2.511462543774956, nar_mse 1.2935168822972978, gradient_norm 0.14860819428799363). Auf 500 Steps etwa 1,7 h Ersparnis. Im Produktions-Baum ungestaged neben den übrigen Patches.

### 12.6 Validierung: Reproduzierbarkeit, DSW bis Step 2, Speicher bei AR 18478 (2026-10-07)

`perf-logs/validate-step2.sh`: drei 2-Step-Läufe, Seed 42, Produktions-Baum. Step 2 ist das lange Item (AR 18478).

- A1/A2 (DSW an, zweimal): Step 1 und Step 2 bitidentisch zwischen den Läufen. Step 2: ar_ce 2.691771518077239, nar_mse 1.4569505576434769, gradient_norm 0.17919589337097158; entspricht dem Baseline-Step-2 aus 8.2 (2.691772 / 1.456951 / 0.179196). Baseline ist reproduzierbar; die Abweichung des `K32=0`-Kontrolllaufs in 8.2/9.2 kam also von einem anderen Rechenpfad, nicht von Nichtdeterminismus.
- B (`DSW=0`): Step 1 und 2 identisch zu A1, also DSW exakt auch nach dem ersten Adam-Update.
- Zeit Step 2 (AR 18478): 272,8 s (DSW=0) -> 257,3 s (DSW an, -15,5 s, -5,7 %). Step 1: 245,3 -> 232,5 s.
- Speicher (peak memory footprint, Maximum über beide Steps): 13,5 GB (DSW=0) gegen 22,7 GB (DSW an). Scratch Step 1 3664 MiB, Step 2 5211 MiB; der alte Puffer bleibt beim Wachsen liegen (3,7 + 5,2 GB), daher +9,1 GB. Metal-Arbeitsmenge ca. 55,7 GB. Maßnahme: Scratch über den Graph-Allokator statt als statischer Puffer (offen, siehe 12.7).

### 12.7 DSW-Scratch über den Graph-Allokator (2026-10-07)

Der dS-Scratch ist kein statischer Metal-Puffer mehr, sondern liegt im eigenen Ziel-Puffer der Op (hinter dQ|dK|dV und delta). `ggml_backend_metal_buffer_type_get_alloc_size` reserviert ihn über `ggml_metal_op_flash_attn_train_back_extra_dsw`; Größe und Aktivierungsbedingungen kommen aus einem gemeinsamen Plan (`hs_fa_dsw_plan_get`), den Alloc-Größe und Encoder teilen (Assert im Encoder). Der Allokator kann den Bereich nach der Op wiederverwenden; es gibt keinen wachsenden Puffer mehr. Patch-Skript: `dsw-alloc-scratch.py`.

Validierung (`fattn-train-test --dsw-check`, `validate-step2.sh`, Produktions-Baum, Seed 42):

- dsw-check: alle Fälle bitweise PASS (inkl. NAR-Hybrid).
- Step 1 und 2 bitidentisch zu DSW=0 und zu 12.6 (ar_ce 2.511462543774956 / 2.691771518077239, nar_mse 1.2935168822972978 / 1.4569505576434769, gradient_norm Step 1 0.14860819428799363, Step 2 0.17919589337097158).
- Zeit: Step 1 232,96 / 232,58 s, Step 2 257,45 / 257,37 s (DSW an) gegen 245,2 / 273,0 s (DSW=0); unverändert zu 12.6.
- Peak memory footprint: 13,54 GB (DSW an, beide Läufe) gegen 13,54 GB (DSW=0); vorher 22,7 GB. DSW kostet damit keinen zusätzlichen Spitzenspeicher mehr.
