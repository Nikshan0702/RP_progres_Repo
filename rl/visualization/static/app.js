/**
 * rl/visualization/static/app.js
 * ==============================
 * Interactive Frontend Controller for the DAPN RL Audio Synthesis Studio.
 * Zero external library dependencies (pure Vanilla JS, Canvas, and SVG).
 */

(function () {
  'use strict';

  // Application State
  const state = {
    mode: 'real', // 'real' or 'demo'
    hasRealAgent: true,
    isInitialized: false,
    isTerminated: false,
    currentStep: 0,
    maxSteps: 8,
    initialLoss: 0,
    currentLoss: 0,
    cumulativeReward: 0,
    improvementPct: 0,
    stages: [],
    stepsHistory: [],
    lossHistory: [],
    rewardHistory: [],
    currentChain: [],
    selectedStageId: 'upload',
    presets: [],
    currentPresetId: 'lead_guitar',
    audioPeaks: {
      dry: null,
      wet: null,
      output: null,
    },
    playback: {
      activeTrack: null,
      isPlaying: false,
      currentTime: 0,
      duration: 0.5,
    },
  };

  // DOM Elements Cache
  const els = {
    presetSelect: document.getElementById('preset-select'),
    btnModeReal: document.getElementById('btn-mode-real'),
    btnModeDemo: document.getElementById('btn-mode-demo'),
    modeBanner: document.getElementById('mode-banner'),
    bannerBadge: document.getElementById('banner-badge'),
    bannerText: document.getElementById('banner-text'),
    metaDevice: document.getElementById('meta-device'),
    btnStep: document.getElementById('btn-step'),
    btnRunAll: document.getElementById('btn-run-all'),
    btnReset: document.getElementById('btn-reset'),
    workflowRibbon: document.getElementById('workflow-ribbon'),
    stageCallout: document.getElementById('stage-detail-callout'),
    calloutTitle: document.getElementById('callout-title'),
    calloutStatus: document.getElementById('callout-status'),
    calloutMessage: document.getElementById('callout-message'),
    calloutIcon: document.getElementById('callout-icon'),
    kpiInitialLoss: document.getElementById('kpi-initial-loss'),
    kpiCurrentLoss: document.getElementById('kpi-current-loss'),
    kpiDeltaLoss: document.getElementById('kpi-delta-loss'),
    kpiImprovement: document.getElementById('kpi-improvement'),
    kpiReward: document.getElementById('kpi-reward'),
    kpiChainLen: document.getElementById('kpi-chain-len'),
    kpiStepCount: document.getElementById('kpi-step-count'),
    actionChainFlow: document.getElementById('action-chain-flow'),
    chainPlaceholder: document.getElementById('chain-placeholder'),
    chainStatusTag: document.getElementById('chain-status-tag'),
    lossChart: document.getElementById('loss-chart'),
    rewardChart: document.getElementById('reward-chart'),
    lossTooltip: document.getElementById('loss-tooltip'),
    rewardTooltip: document.getElementById('reward-tooltip'),
    btnGlobalPlay: document.getElementById('btn-global-play'),
    globalPlayIcon: document.getElementById('global-play-icon'),
    btnDownloadOutput: document.getElementById('btn-download-output'),
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    actionDistBars: document.getElementById('action-distribution-bars'),
    spectralBinsVis: document.getElementById('spectral-bins-vis'),
    historySlotsList: document.getElementById('history-slots-list'),
    progressMetricsList: document.getElementById('progress-metrics-list'),
    rewardTableWrapper: document.getElementById('reward-table-wrapper'),
    jsonCode: document.getElementById('json-code'),
    btnCopyJson: document.getElementById('btn-copy-json'),
    uploadModal: document.getElementById('upload-modal'),
    btnModalClose: document.getElementById('btn-modal-close'),
    btnModalCancel: document.getElementById('btn-modal-cancel'),
    btnModalSubmit: document.getElementById('btn-modal-submit'),
    inputFileDry: document.getElementById('input-file-dry'),
    inputFileWet: document.getElementById('input-file-wet'),
    dryUploadLabel: document.getElementById('dry-upload-label'),
    wetUploadLabel: document.getElementById('wet-upload-label'),
    audioPlayers: {
      dry: document.getElementById('audio-player-dry'),
      wet: document.getElementById('audio-player-wet'),
      output: document.getElementById('audio-player-output'),
    },
    playheads: {
      dry: document.getElementById('playhead-dry'),
      wet: document.getElementById('playhead-wet'),
      output: document.getElementById('playhead-output'),
    },
    timeDisplays: {
      dry: document.getElementById('time-dry'),
      wet: document.getElementById('time-wet'),
      output: document.getElementById('time-output'),
    },
    waveforms: {
      dry: document.getElementById('waveform-dry'),
      wet: document.getElementById('waveform-wet'),
      output: document.getElementById('waveform-output'),
    },
    trackPlayBtns: document.querySelectorAll('.track-play-btn'),
  };

  // Upload modal cache
  let uploadedDryBase64 = null;
  let uploadedWetBase64 = null;

  // =========================================================================
  // API Calls
  // =========================================================================
  // Errors from the backend (e.g. 503 real backend unavailable) are surfaced, never turned into demo data.
  async function parseResponse(res) {
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON error body */ }
    if (!res.ok) {
      throw new Error((data && data.error) || `HTTP ${res.status}`);
    }
    return data;
  }

  function showBackendUnavailable(reason) {
    els.modeBanner.className = 'mode-banner demo';
    els.bannerBadge.textContent = 'REAL BACKEND UNAVAILABLE';
    els.bannerText.innerHTML = 'Real RL execution cannot run: <b>' + String(reason || 'unknown reason').replace(/[<>&]/g, '') +
      '</b>. Nothing is substituted; choose <b>Demo Mode</b> explicitly for scripted, SIMULATED output.';
  }

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      state.hasRealAgent = data.has_real_agent;
      state.presets = data.presets || [];
      if (!state.hasRealAgent) {
        showBackendUnavailable(data.backend_error);   // no silent switch to demo
      }
      return data;
    } catch (e) {
      console.warn('Status check failed:', e);
    }
  }

  async function apiInitialize(params = {}) {
    setLoading(true);
    try {
      const payload = {
        mode: state.mode,
        max_steps: state.maxSteps,
        ...params,
      };
      const res = await fetch('/api/initialize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await parseResponse(res);
      updateStateFromSummary(data);
    } catch (e) {
      console.error('Initialize failed:', e);
      alert('Initialization error: ' + e.message);
      if (state.mode === 'real') showBackendUnavailable(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function apiStep() {
    setLoading(true);
    try {
      const res = await fetch('/api/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await parseResponse(res);
      updateStateFromSummary(data);
    } catch (e) {
      console.error('Step failed:', e);
      alert('Step error: ' + e.message);
    } finally {
      setLoading(false);
    }
  }

  async function apiRunAll() {
    setLoading(true);
    try {
      const res = await fetch('/api/run_all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await parseResponse(res);
      updateStateFromSummary(data);
    } catch (e) {
      console.error('Run all failed:', e);
      alert('Run all error: ' + e.message);
    } finally {
      setLoading(false);
    }
  }

  async function apiReset() {
    setLoading(true);
    try {
      const res = await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await parseResponse(res);
      updateStateFromSummary(data);
      // Re-initialize with currently selected preset
      initCurrentPreset();
    } catch (e) {
      console.error('Reset failed:', e);
    } finally {
      setLoading(false);
    }
  }

  function initCurrentPreset() {
    const preset = state.presets.find((p) => p.id === state.currentPresetId);
    if (preset) {
      apiInitialize({
        dry_path: preset.dry_path,
        wet_path: preset.wet_path,
      });
    } else {
      apiInitialize();
    }
  }

  // =========================================================================
  // State Update & UI Re-rendering
  // =========================================================================
  function updateStateFromSummary(summary) {
    if (!summary) return;

    state.mode = summary.mode;
    state.isInitialized = summary.is_initialized;
    state.isTerminated = summary.is_terminated;
    state.currentStep = summary.current_step;
    state.maxSteps = summary.max_steps;
    state.simulated = !!summary.simulated;
    state.baselineName = summary.baseline_name || 'dry';
    state.dryLoss = summary.dry_loss;
    state.predictorLoss = summary.predictor_loss;
    state.initialLoss = summary.baseline_loss !== undefined && summary.baseline_loss !== null ? summary.baseline_loss : summary.initial_loss;
    state.currentLoss = summary.current_loss;
    state.cumulativeReward = summary.cumulative_reward;
    state.improvementPct = summary.improvement_pct;
    state.stages = summary.stages || [];
    state.stepsHistory = summary.steps_history || [];
    state.lossHistory = summary.loss_history || [];
    state.rewardHistory = summary.reward_history || [];
    state.currentChain = summary.current_chain || [];

    // Extract audio peaks if available in stages
    const uploadStage = state.stages.find((s) => s.id === 'upload');
    if (uploadStage && uploadStage.data) {
      state.audioPeaks.dry = uploadStage.data.dry_peaks;
      state.audioPeaks.wet = uploadStage.data.wet_peaks;
      if (uploadStage.data.dry_metrics) {
        state.playback.duration = uploadStage.data.dry_metrics.duration_sec || 0.5;
        document.getElementById('metrics-dry').textContent = `${uploadStage.data.dry_metrics.sample_rate}Hz • ${uploadStage.data.dry_metrics.peak_db} dB`;
        document.getElementById('metrics-wet').textContent = `${uploadStage.data.wet_metrics.sample_rate}Hz • ${uploadStage.data.wet_metrics.peak_db} dB`;
      }
    }

    const outputStage = state.stages.find((s) => s.id === 'output');
    if (outputStage && outputStage.status === 'completed' && outputStage.data) {
      state.audioPeaks.output = outputStage.data.final_peaks;
      if (outputStage.data.final_metrics) {
        document.getElementById('metrics-output').textContent = `${outputStage.data.final_metrics.sample_rate}Hz • ${outputStage.data.final_metrics.peak_db} dB`;
      }
      const hasAudio = !summary.simulated && summary.has_output_audio;
      els.btnDownloadOutput.disabled = !hasAudio;
      document.querySelector('[data-track="output"]').disabled = !hasAudio;

      // Set audio sources for playback
      if (outputStage.data.base64_audio) {
        els.audioPlayers.output.src = outputStage.data.base64_audio;
      }
    } else {
      state.audioPeaks.output = null;
      els.btnDownloadOutput.disabled = true;
      document.querySelector('[data-track="output"]').disabled = true;
    }

    // Set dry and wet audio sources via API streaming
    els.audioPlayers.dry.src = '/api/audio?type=dry';
    els.audioPlayers.wet.src = '/api/audio?type=wet';
    if (!els.audioPlayers.output.src && state.isTerminated && !summary.simulated && summary.has_output_audio) {
      els.audioPlayers.output.src = '/api/audio?type=output';
    }

    renderAll();
  }

  function renderAll() {
    renderKPIs();
    renderWorkflowRibbon();
    renderActionChainGraph();
    renderLossChart();
    renderRewardChart();
    renderWaveforms();
    renderInspectorTabs();
    updateButtonStates();
  }

  function renderKPIs() {
    const lbl = document.getElementById('kpi-initial-label');
    if (lbl) {
      lbl.textContent = (state.baselineName === 'predictor' ? 'Start Loss (Predictor-only)' : 'Initial Loss (Dry)') + (state.simulated ? ' [SIMULATED]' : '');
    }
    els.kpiInitialLoss.title = `dry ${state.dryLoss}; predictor ${state.predictorLoss}`;
    els.kpiInitialLoss.textContent = state.initialLoss < 900 ? state.initialLoss.toFixed(4) : '--';
    els.kpiCurrentLoss.textContent = state.currentLoss < 900 ? state.currentLoss.toFixed(4) : '--';

    const delta = state.initialLoss < 900 ? state.initialLoss - state.currentLoss : 0;
    els.kpiDeltaLoss.textContent = `Δ ${delta >= 0 ? '-' : '+'}${Math.abs(delta).toFixed(4)}`;

    els.kpiImprovement.textContent = `${state.improvementPct >= 0 ? '+' : ''}${state.improvementPct.toFixed(1)}%`;  // signed: negative = degradation
    els.kpiReward.textContent = `${state.cumulativeReward >= 0 ? '+' : ''}${state.cumulativeReward.toFixed(3)}`;
    els.kpiChainLen.textContent = `${state.currentChain.length} / ${state.maxSteps}`;
    els.kpiStepCount.textContent = `Step ${state.currentStep} ${state.isTerminated ? '(Finished)' : ''}`;

    els.chainStatusTag.textContent = state.isTerminated
      ? `Completed (${state.currentChain.length} Effects)`
      : state.currentChain.length > 0
      ? `Active (${state.currentChain.length} Effects)`
      : 'Chain Empty';
  }

  // =========================================================================
  // Section 1: Complete Workflow Progression Ribbon (Requirement 2)
  // =========================================================================
  function renderWorkflowRibbon() {
    els.workflowRibbon.innerHTML = '';

    const stageIcons = {
      upload: '📤',
      preprocess: '🎛️',
      state: '🧠',
      action: '🎯',
      chain: '🔗',
      dsp: '⚡',
      loss: '📉',
      reward: '🏆',
      next: '🔄',
      stop: '🛑',
      output: '🔊',
    };

    state.stages.forEach((stage, idx) => {
      const isLast = idx === state.stages.length - 1;
      const wrapper = document.createElement('div');
      wrapper.className = 'stage-node-wrapper';

      const node = document.createElement('div');
      node.className = `stage-node ${stage.status}`;
      if (stage.id === state.selectedStageId) {
        node.classList.add('selected');
      }

      node.innerHTML = `
        <div class="stage-icon-circle">${stageIcons[stage.id] || '⚙️'}</div>
        <div class="stage-labels">
          <span class="stage-name">${stage.name}</span>
          <span class="stage-status-pill ${stage.status}">${stage.status}</span>
        </div>
      `;

      node.addEventListener('click', () => {
        selectStage(stage.id);
      });

      wrapper.appendChild(node);

      if (!isLast) {
        const connector = document.createElement('div');
        connector.className = `stage-connector ${stage.status === 'completed' ? 'completed' : ''}`;
        wrapper.appendChild(connector);
      }

      els.workflowRibbon.appendChild(wrapper);
    });

    renderStageCallout();
  }

  function selectStage(stageId) {
    state.selectedStageId = stageId;
    renderWorkflowRibbon();
  }

  function renderStageCallout() {
    const stage = state.stages.find((s) => s.id === state.selectedStageId);
    if (!stage) return;

    els.calloutTitle.textContent = `${stage.title} (${stage.name})`;
    els.calloutStatus.textContent = stage.status;
    els.calloutStatus.className = `callout-status-pill ${stage.status}`;
    els.calloutMessage.textContent = stage.message || stage.description;
  }

  // =========================================================================
  // Section 2: Action Graph & Effect Chain Sequence (Requirement 3 - Main Focus)
  // =========================================================================
  function renderActionChainGraph() {
    // Keep the starting dry input node
    const nodes = els.actionChainFlow.querySelectorAll('.chain-node:not(.start-node)');
    nodes.forEach((n) => n.remove());

    const arrows = els.actionChainFlow.querySelectorAll('.chain-arrow');
    arrows.forEach((a) => a.remove());

    if (state.stepsHistory.length === 0) {
      els.chainPlaceholder.style.display = 'flex';
      return;
    }

    els.chainPlaceholder.style.display = 'none';

    const effectIcons = {
      EQ: '🎚️',
      Compressor: '🗜️',
      Reverb: '🌊',
      Distortion: '⚡',
      Delay: '⏱️',
      Chorus: '💫',
      Flanger: '🌀',
      STOP: '🛑',
    };

    state.stepsHistory.forEach((step, idx) => {
      // Connecting Arrow
      const arrow = document.createElement('div');
      arrow.className = 'chain-arrow';
      arrow.innerHTML = '➔';
      els.actionChainFlow.appendChild(arrow);

      // Effect / Action Node
      const node = document.createElement('div');
      node.className = `chain-node ${step.is_stop ? 'stop-node' : ''}`;

      if (step.is_stop) {
        node.innerHTML = `
          <div class="node-badge">
            <span class="step-chip">STEP ${step.step}</span>
            <span class="prob-chip">TERMINATION</span>
          </div>
          <div class="node-icon">🛑</div>
          <div class="node-name">STOP Action</div>
          <div class="node-meta">Chain Finalized</div>
          <div style="font-size: 11px; color: var(--accent-emerald); margin-top: 8px;">
            Reward: ${step.step_reward >= 0 ? '+' : ''}${step.step_reward.toFixed(3)}
          </div>
        `;
      } else {
        // Build Parameter Rows
        let paramsHtml = '';
        if (step.parameters && step.parameters.length > 0) {
          paramsHtml = `
            <div class="node-params-table">
              ${step.parameters
                .map((p) => {
                  const deltaClass = p.delta_normalized > 0 ? 'pos' : p.delta_normalized < 0 ? 'neg' : 'zero';
                  const deltaStr = p.prev_normalized !== null ? (p.delta_normalized >= 0 ? `+${p.delta_normalized}` : `${p.delta_normalized}`) : 'Init';
                  const pct = Math.round(p.normalized * 100);
                  return `
                    <div class="param-row">
                      <div class="param-header">
                        <span class="param-name">${p.name}</span>
                        <div class="param-values">
                          <span class="param-phys">${p.display}</span>
                          <span class="param-delta ${deltaClass}">${deltaStr}</span>
                        </div>
                      </div>
                      <div class="param-bar-bg">
                        <div class="param-bar-fill" style="width: ${pct}%;"></div>
                      </div>
                    </div>
                  `;
                })
                .join('')}
            </div>
          `;
        }

        const probText = step.action_diagnostics
          ? `P=${step.action_diagnostics.distribution[step.discrete_action_idx]?.percentage || '--'}%`
          : `Step ${step.step}`;

        node.innerHTML = `
          <div class="node-badge">
            <span class="step-chip">STEP ${step.step}</span>
            <span class="prob-chip">${probText}</span>
          </div>
          <div class="node-title-row">
            <div class="node-icon">${effectIcons[step.action] || '🎛️'}</div>
            <div>
              <div class="node-name">${step.action}</div>
              <div class="node-meta">Loss: ${step.current_loss.toFixed(4)} (Δ ${step.delta_loss >= 0 ? '-' : '+'}${Math.abs(step.delta_loss).toFixed(4)})</div>
            </div>
          </div>
          ${paramsHtml}
        `;
      }

      els.actionChainFlow.appendChild(node);
    });

    // If terminated, add output node
    if (state.isTerminated) {
      const arrow = document.createElement('div');
      arrow.className = 'chain-arrow';
      arrow.innerHTML = '➔';
      els.actionChainFlow.appendChild(arrow);

      const outNode = document.createElement('div');
      outNode.className = 'chain-node start-node';
      outNode.style.background = 'rgba(16, 185, 129, 0.08)';
      outNode.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      outNode.innerHTML = `
        <div class="node-badge">OUTPUT</div>
        <div class="node-icon">🔊</div>
        <div class="node-name">Final Audio</div>
        <div class="node-meta">${state.improvementPct >= 0 ? "+" : ""}${state.improvementPct.toFixed(1)}% vs ${state.baselineName}</div>
      `;
      els.actionChainFlow.appendChild(outNode);
    }
  }

  // =========================================================================
  // Section 3: Interactive Charts: Loss & Reward (Requirement 4 - Main Focus)
  // =========================================================================
  function renderLossChart() {
    const svg = els.lossChart;
    svg.innerHTML = '';

    if (state.lossHistory.length === 0) {
      svg.innerHTML = `<text x="300" y="120" text-anchor="middle" fill="#64748b" font-size="12">Awaiting workflow initialization...</text>`;
      return;
    }

    const width = 600;
    const height = 240;
    const padding = { top: 20, right: 30, bottom: 35, left: 50 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Extents
    const losses = state.lossHistory.map((d) => d.loss);
    const maxLoss = Math.max(...losses, state.initialLoss, 1.0) * 1.1;
    const minLoss = 0.0;
    const maxSteps = Math.max(state.lossHistory.length - 1, 4);

    const getX = (step) => padding.left + (step / maxSteps) * chartW;
    const getY = (loss) => padding.top + chartH - ((loss - minLoss) / (maxLoss - minLoss)) * chartH;

    // Grid Lines & Y-axis Ticks
    const numTicks = 5;
    for (let i = 0; i <= numTicks; i++) {
      const val = minLoss + (i / numTicks) * (maxLoss - minLoss);
      const y = getY(val);
      svg.innerHTML += `
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="chart-grid-line" />
        <text x="${padding.left - 8}" y="${y + 3}" text-anchor="end" class="chart-axis-text">${val.toFixed(1)}</text>
      `;
    }

    // X-axis Ticks
    for (let s = 0; s <= maxSteps; s++) {
      const x = getX(s);
      svg.innerHTML += `
        <text x="${x}" y="${height - 12}" text-anchor="middle" class="chart-axis-text">${s === 0 ? 'Dry' : 'S' + s}</text>
      `;
    }

    // Target Threshold Line (0.40)
    const targetY = getY(0.40);
    svg.innerHTML += `
      <line x1="${padding.left}" y1="${targetY}" x2="${width - padding.right}" y2="${targetY}" stroke="var(--accent-rose)" stroke-dasharray="4,4" stroke-width="1.5" />
    `;

    // Baseline Line
    const baselineY = getY(state.initialLoss);
    svg.innerHTML += `
      <line x1="${padding.left}" y1="${baselineY}" x2="${width - padding.right}" y2="${baselineY}" stroke="#64748b" stroke-dasharray="3,3" stroke-width="1" />
    `;

    // Loss Curve Path
    let pathD = '';
    state.lossHistory.forEach((pt, idx) => {
      const x = getX(pt.step);
      const y = getY(pt.loss);
      if (idx === 0) {
        pathD += `M ${x} ${y}`;
      } else {
        pathD += ` L ${x} ${y}`;
      }
    });

    svg.innerHTML += `<path d="${pathD}" class="chart-curve loss" />`;

    // Data Points
    state.lossHistory.forEach((pt) => {
      const x = getX(pt.step);
      const y = getY(pt.loss);
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', '4');
      circle.setAttribute('fill', 'var(--accent-indigo)');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('class', 'chart-dot');

      circle.addEventListener('mouseenter', (e) => {
        showTooltip(els.lossTooltip, e, `
          <b>${pt.action} (Step ${pt.step})</b><br/>
          MR-STFT Loss: <b>${pt.loss.toFixed(4)}</b><br/>
          Improvement: <b>${pt.improvement_pct.toFixed(1)}%</b>
        `);
      });
      circle.addEventListener('mouseleave', () => hideTooltip(els.lossTooltip));

      svg.appendChild(circle);
    });
  }

  function renderRewardChart() {
    const svg = els.rewardChart;
    svg.innerHTML = '';

    if (state.rewardHistory.length === 0) {
      svg.innerHTML = `<text x="300" y="120" text-anchor="middle" fill="#64748b" font-size="12">Awaiting RL action execution...</text>`;
      return;
    }

    const width = 600;
    const height = 240;
    const padding = { top: 20, right: 30, bottom: 35, left: 50 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const allRewards = state.rewardHistory.map((d) => d.reward);
    const allCums = state.rewardHistory.map((d) => d.cumulative);
    const maxVal = Math.max(...allRewards, ...allCums, 10.0) * 1.15;
    const minVal = Math.min(...allRewards, ...allCums, -5.0) * 1.15;
    const maxSteps = Math.max(state.rewardHistory.length, 4);

    const getX = (step) => padding.left + ((step - 0.5) / maxSteps) * chartW;
    const getY = (val) => padding.top + chartH - ((val - minVal) / (maxVal - minVal)) * chartH;
    const zeroY = getY(0.0);

    // Grid & Zero Line
    svg.innerHTML += `
      <line x1="${padding.left}" y1="${zeroY}" x2="${width - padding.right}" y2="${zeroY}" stroke="rgba(255,255,255,0.2)" stroke-width="1.2" />
      <text x="${padding.left - 8}" y="${zeroY + 3}" text-anchor="end" class="chart-axis-text">0.0</text>
    `;

    // Step Reward Bars
    const barWidth = Math.min(26, (chartW / maxSteps) * 0.45);
    state.rewardHistory.forEach((pt) => {
      const x = getX(pt.step) - barWidth / 2;
      const y = getY(Math.max(0, pt.reward));
      const h = Math.abs(getY(pt.reward) - zeroY);
      const isPos = pt.reward >= 0;

      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', x);
      rect.setAttribute('y', isPos ? getY(pt.reward) : zeroY);
      rect.setAttribute('width', barWidth);
      rect.setAttribute('height', Math.max(2, h));
      rect.setAttribute('rx', '3');
      rect.setAttribute('fill', isPos ? 'var(--accent-emerald)' : 'var(--accent-rose)');
      rect.setAttribute('opacity', '0.85');

      rect.addEventListener('mouseenter', (e) => {
        showTooltip(els.rewardTooltip, e, `
          <b>${pt.action} (Step ${pt.step})</b><br/>
          Step Reward: <b>${pt.reward >= 0 ? '+' : ''}${pt.reward.toFixed(3)}</b><br/>
          Cumulative Return: <b>${pt.cumulative.toFixed(3)}</b>
        `);
      });
      rect.addEventListener('mouseleave', () => hideTooltip(els.rewardTooltip));

      svg.appendChild(rect);
    });

    // Cumulative Polyline
    let cumPath = '';
    state.rewardHistory.forEach((pt, idx) => {
      const x = getX(pt.step);
      const y = getY(pt.cumulative);
      if (idx === 0) cumPath += `M ${x} ${y}`;
      else cumPath += ` L ${x} ${y}`;
    });

    svg.innerHTML += `<path d="${cumPath}" class="chart-curve cumulative" />`;

    // Cumulative Dots
    state.rewardHistory.forEach((pt) => {
      const x = getX(pt.step);
      const y = getY(pt.cumulative);
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', '4');
      circle.setAttribute('fill', 'var(--accent-cyan)');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('class', 'chart-dot');
      svg.appendChild(circle);
    });
  }

  function showTooltip(tooltipEl, event, htmlContent) {
    tooltipEl.innerHTML = htmlContent;
    tooltipEl.classList.remove('hidden');
    const rect = tooltipEl.parentElement.getBoundingClientRect();
    const x = event.clientX - rect.left + 10;
    const y = event.clientY - rect.top - 35;
    tooltipEl.style.left = `${x}px`;
    tooltipEl.style.top = `${y}px`;
  }

  function hideTooltip(tooltipEl) {
    tooltipEl.classList.add('hidden');
  }

  // =========================================================================
  // Section 4: Audio Studio & Waveform Visualizers (Requirement 5)
  // =========================================================================
  function renderWaveforms() {
    drawWaveform('dry', state.audioPeaks.dry, '#3b82f6');
    drawWaveform('wet', state.audioPeaks.wet, '#f59e0b');
    drawWaveform('output', state.audioPeaks.output, '#10b981');
  }

  function drawWaveform(trackId, peaksData, colorHex) {
    const canvas = els.waveforms[trackId];
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Center baseline
    const centerY = h / 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(w, centerY);
    ctx.stroke();

    if (!peaksData || !peaksData.max_peaks || peaksData.max_peaks.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '11px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText(trackId === 'output' ? 'Waveform renders on synthesis' : 'Loading waveform envelope...', w / 2, centerY + 4);
      return;
    }

    const maxPeaks = peaksData.max_peaks;
    const minPeaks = peaksData.min_peaks;
    const numBars = maxPeaks.length;
    const barWidth = Math.max(1, w / numBars);

    ctx.fillStyle = colorHex;
    for (let i = 0; i < numBars; i++) {
      const x = i * barWidth;
      const topH = Math.max(1, maxPeaks[i] * (centerY - 4));
      const btmH = Math.max(1, Math.abs(minPeaks[i]) * (centerY - 4));

      // Draw top and bottom envelope bar
      ctx.fillRect(x, centerY - topH, barWidth - 0.5, topH);
      ctx.fillRect(x, centerY, barWidth - 0.5, btmH);
    }
  }

  // Audio Playback Handling
  function setupAudioPlayback() {
    ['dry', 'wet', 'output'].forEach((trackId) => {
      const player = els.audioPlayers[trackId];
      const playhead = els.playheads[trackId];
      const timeDisplay = els.timeDisplays[trackId];
      const canvas = els.waveforms[trackId];

      player.addEventListener('timeupdate', () => {
        const cur = player.currentTime;
        const dur = player.duration || state.playback.duration || 1;
        const pct = (cur / dur) * 100;
        playhead.style.left = `${pct}%`;
        timeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
      });

      player.addEventListener('ended', () => {
        playhead.style.left = '0%';
        updatePlayBtnIcon(trackId, false);
      });

      // Canvas click to seek
      canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const pct = clickX / rect.width;
        const dur = player.duration || state.playback.duration || 1;
        player.currentTime = pct * dur;
        playhead.style.left = `${pct * 100}%`;
      });
    });

    // Track play buttons
    els.trackPlayBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const trackId = btn.getAttribute('data-track');
        toggleTrackPlay(trackId);
      });
    });

    // Global Play All Synced
    els.btnGlobalPlay.addEventListener('click', () => {
      const isAnyPlaying = Object.values(els.audioPlayers).some((p) => !p.paused);
      if (isAnyPlaying) {
        Object.values(els.audioPlayers).forEach((p) => p.pause());
        els.globalPlayIcon.textContent = '▶';
      } else {
        Object.values(els.audioPlayers).forEach((p) => {
          if (p.src) {
            p.currentTime = 0;
            p.play().catch(() => {});
          }
        });
        els.globalPlayIcon.textContent = '⏸';
      }
    });

    // Download Output WAV
    els.btnDownloadOutput.addEventListener('click', () => {
      window.location.href = '/api/download?type=output';
    });
  }

  function toggleTrackPlay(trackId) {
    const player = els.audioPlayers[trackId];
    if (!player || !player.src) return;

    if (player.paused) {
      // Pause other tracks
      Object.keys(els.audioPlayers).forEach((tid) => {
        if (tid !== trackId) {
          els.audioPlayers[tid].pause();
          updatePlayBtnIcon(tid, false);
        }
      });
      player.play().then(() => updatePlayBtnIcon(trackId, true)).catch((e) => console.warn(e));
    } else {
      player.pause();
      updatePlayBtnIcon(trackId, false);
    }
  }

  function updatePlayBtnIcon(trackId, isPlaying) {
    const btn = document.querySelector(`.track-play-btn[data-track="${trackId}"]`);
    if (btn) btn.textContent = isPlaying ? '⏸' : '▶';
  }

  function formatTime(sec) {
    const s = Math.floor(sec);
    const ms = Math.floor((sec - s) * 10);
    return `00:0${s}.${ms}`;
  }

  // =========================================================================
  // Section 5: Deep Diagnostic Inspector (Tabs)
  // =========================================================================
  function renderInspectorTabs() {
    renderTabPolicy();
    renderTabState();
    renderTabRewards();
    renderTabJson();
  }

  function renderTabPolicy() {
    const lastStep = state.stepsHistory[state.stepsHistory.length - 1];
    if (!lastStep || !lastStep.action_diagnostics) {
      els.actionDistBars.innerHTML = `<div style="color: var(--text-muted); font-size: 12px;">Execute an action to view policy categorical distribution.</div>`;
      return;
    }

    const dist = lastStep.action_diagnostics.distribution || [];
    els.actionDistBars.innerHTML = dist
      .map((item) => {
        const isSelected = item.is_selected;
        const isMasked = !item.is_valid;
        return `
          <div class="dist-bar-card ${isSelected ? 'selected' : ''} ${isMasked ? 'masked' : ''}">
            <div class="dist-bar-header">
              <span>${item.action}</span>
              <span>${item.percentage}%</span>
            </div>
            <div class="dist-bar-bg">
              <div class="dist-bar-fill" style="width: ${item.percentage}%;"></div>
            </div>
            <div class="dist-bar-meta">
              ${isMasked ? `🔒 Masked: ${item.mask_reason}` : isSelected ? '✓ Selected' : `Logit: ${item.logit}`}
            </div>
          </div>
        `;
      })
      .join('');
  }

  function renderTabState() {
    // 1. Spectral visualizer bins
    const binsContainer = els.spectralBinsVis;
    binsContainer.innerHTML = '';
    for (let i = 0; i < 32; i++) {
      const h = Math.max(4, Math.floor(Math.sin(i / 5) * 25 + 30));
      binsContainer.innerHTML += `<div class="spectral-bin" style="height: ${h}px;"></div>`;
    }

    // 2. History Slots
    els.historySlotsList.innerHTML = state.currentChain
      .map(
        (item, idx) => `
        <div class="history-slot-item">
          <span>Slot ${idx + 1}: <b>${item.effect}</b></span>
          <span style="color: var(--accent-emerald);">Active</span>
        </div>
      `
      )
      .join('');
    if (state.currentChain.length === 0) {
      els.historySlotsList.innerHTML = `<div style="font-size: 11px; color: var(--text-muted);">Slots 1-8: [EMPTY]</div>`;
    }

    // 3. Progress metrics
    els.progressMetricsList.innerHTML = `
      <div class="history-slot-item"><span>Step Progress:</span> <b>${Math.round((state.currentStep / state.maxSteps) * 100)}%</b></div>
      <div class="history-slot-item"><span>Chain Occupancy:</span> <b>${state.currentChain.length} / ${state.maxSteps}</b></div>
      <div class="history-slot-item"><span>Current STFT Loss:</span> <b>${state.currentLoss < 900 ? state.currentLoss.toFixed(4) : '--'}</b></div>
      <div class="history-slot-item"><span>Improvement Ratio:</span> <b>${(state.improvementPct / 100).toFixed(3)}</b></div>
    `;
  }

  function renderTabRewards() {
    if (state.stepsHistory.length === 0) {
      els.rewardTableWrapper.innerHTML = `<div style="color: var(--text-muted); font-size: 12px;">No reward breakdown available yet.</div>`;
      return;
    }

    const tableRows = state.stepsHistory
      .map((s) => {
        const rb = s.reward_breakdown || {};
        return `
        <tr style="border-bottom: 1px solid var(--border-subtle); font-size: 11.5px; font-family: var(--font-mono);">
          <td style="padding: 6px 10px;">Step ${s.step}</td>
          <td style="padding: 6px 10px; font-weight: 700; color: #fff;">${s.action}</td>
          <td style="padding: 6px 10px; color: ${s.delta_loss >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${s.delta_loss >= 0 ? '-' : '+'}${Math.abs(s.delta_loss).toFixed(4)}</td>
          <td style="padding: 6px 10px;">${rb.loss_improvement_reward ? '+' + rb.loss_improvement_reward.toFixed(3) : '--'}</td>
          <td style="padding: 6px 10px; color: var(--accent-rose);">${rb.step_penalty ? rb.step_penalty.toFixed(3) : '-0.050'}</td>
          <td style="padding: 6px 10px; color: var(--accent-amber);">${rb.terminal_bonus ? '+' + rb.terminal_bonus.toFixed(3) : '0.000'}</td>
          <td style="padding: 6px 10px; font-weight: 700; color: ${s.step_reward >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${s.step_reward >= 0 ? '+' : ''}${s.step_reward.toFixed(3)}</td>
        </tr>
      `;
      })
      .join('');

    els.rewardTableWrapper.innerHTML = `
      <table style="width: 100%; text-align: left; border-collapse: collapse;">
        <thead>
          <tr style="color: var(--text-muted); font-size: 11px; border-bottom: 1px solid var(--border-subtle);">
            <th style="padding: 6px 10px;">Step</th>
            <th style="padding: 6px 10px;">Action</th>
            <th style="padding: 6px 10px;">Δ Loss</th>
            <th style="padding: 6px 10px;">10&times;ΔL Reward</th>
            <th style="padding: 6px 10px;">Step Penalty</th>
            <th style="padding: 6px 10px;">Terminal Bonus</th>
            <th style="padding: 6px 10px;">Step Return</th>
          </tr>
        </thead>
        <tbody>${tableRows}</tbody>
      </table>
    `;
  }

  function renderTabJson() {
    const chainDict = { chain: state.currentChain };
    els.jsonCode.textContent = JSON.stringify(chainDict, null, 2);
  }

  // =========================================================================
  // Controls & Event Handlers
  // =========================================================================
  function setupControls() {
    // Mode toggles
    els.btnModeReal.addEventListener('click', () => setMode('real'));
    els.btnModeDemo.addEventListener('click', () => setMode('demo'));

    // Stepping and Running
    els.btnStep.addEventListener('click', apiStep);
    els.btnRunAll.addEventListener('click', apiRunAll);
    els.btnReset.addEventListener('click', apiReset);

    // Preset selection
    els.presetSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'custom') {
        openUploadModal();
      } else {
        state.currentPresetId = val;
        initCurrentPreset();
      }
    });

    // Inspector Tabs
    els.tabButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        els.tabButtons.forEach((b) => b.classList.remove('active'));
        els.tabContents.forEach((c) => c.classList.remove('active'));
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        document.getElementById(targetId)?.classList.add('active');
      });
    });

    // Copy JSON
    els.btnCopyJson.addEventListener('click', () => {
      navigator.clipboard.writeText(els.jsonCode.textContent).then(() => {
        const orig = els.btnCopyJson.textContent;
        els.btnCopyJson.textContent = '✓ Copied!';
        setTimeout(() => (els.btnCopyJson.textContent = orig), 1500);
      });
    });

    // Modal controls
    els.btnModalClose.addEventListener('click', closeUploadModal);
    els.btnModalCancel.addEventListener('click', closeUploadModal);
    els.inputFileDry.addEventListener('change', handleDryFileSelect);
    els.inputFileWet.addEventListener('change', handleWetFileSelect);
    els.btnModalSubmit.addEventListener('click', handleUploadSubmit);
  }

  function setMode(newMode) {
    state.mode = newMode;
    if (newMode === 'demo') {
      els.btnModeReal.classList.remove('active');
      els.btnModeDemo.classList.add('active');
      els.modeBanner.className = 'mode-banner demo';
      els.bannerBadge.textContent = 'DEMO / SIMULATED';
      els.bannerText.innerHTML =
        'Running in <b>DEMO MODE</b>. Simulated progression for inspecting visualization components without live PyTorch execution.';
    } else {
      els.btnModeDemo.classList.remove('active');
      els.btnModeReal.classList.add('active');
      els.modeBanner.className = 'mode-banner real';
      els.bannerBadge.textContent = 'REAL RL POLICY';
      els.bannerText.innerHTML =
        'Executing PyTorch Differentiable Audio Processing Network (DAPN) with trained GRRL checkpoint (<code>checkpoints_v2/grrl/best_grrl_model.pth</code>) and Multi-Resolution STFT Loss.';
    }
    // Re-initialize session in selected mode
    initCurrentPreset();
  }

  function updateButtonStates() {
    const disabled = state.isTerminated || state.currentStep >= state.maxSteps;
    els.btnStep.disabled = disabled;
    els.btnRunAll.disabled = disabled;
  }

  function setLoading(isLoading) {
    if (isLoading) {
      els.btnStep.disabled = true;
      els.btnRunAll.disabled = true;
    } else {
      updateButtonStates();
    }
  }

  // Upload Handling
  function openUploadModal() {
    els.uploadModal.classList.remove('hidden');
  }

  function closeUploadModal() {
    els.uploadModal.classList.add('hidden');
    els.presetSelect.value = state.currentPresetId;
  }

  function handleDryFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    els.dryUploadLabel.textContent = file.name;
    const reader = new FileReader();
    reader.onload = (evt) => {
      uploadedDryBase64 = evt.target.result;
      checkUploadReady();
    };
    reader.readAsDataURL(file);
  }

  function handleWetFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    els.wetUploadLabel.textContent = file.name;
    const reader = new FileReader();
    reader.onload = (evt) => {
      uploadedWetBase64 = evt.target.result;
      checkUploadReady();
    };
    reader.readAsDataURL(file);
  }

  function checkUploadReady() {
    els.btnModalSubmit.disabled = !(uploadedDryBase64 && uploadedWetBase64);
  }

  async function handleUploadSubmit() {
    if (!uploadedDryBase64 || !uploadedWetBase64) return;
    setLoading(true);
    closeUploadModal();
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dry_base64: uploadedDryBase64,
          wet_base64: uploadedWetBase64,
          mode: state.mode,
        }),
      });
      const data = await parseResponse(res);
      updateStateFromSummary(data);
    } catch (e) {
      console.error('Upload failed:', e);
      alert('Upload failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  }

  // =========================================================================
  // Initialization
  // =========================================================================
  async function init() {
    setupControls();
    setupAudioPlayback();
    const statusData = await fetchStatus();
    if (statusData && statusData.is_initialized) {
      try {
        const res = await fetch('/api/summary');
        const data = await res.json();
        updateStateFromSummary(data);
      } catch (e) {
        initCurrentPreset();
      }
    } else {
      initCurrentPreset();
    }
  }

  // Start when DOM is loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
