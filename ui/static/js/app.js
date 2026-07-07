const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

function stageMeta(stage) {
  if (!stage) return '';
  const ms = stage.latency_ms != null ? `${stage.latency_ms} ms` : '';
  return ms ? `<div class="meta">${ms}</div>` : '';
}

function renderAddress(stage) {
  if (!stage) return;
  if (!stage.ok) {
    $('card-address').innerHTML = `<div class="error">${esc(stage.error)}</div>${stageMeta(stage)}`;
    return;
  }
  const a = stage.value || {};
  const rows = ['street', 'city', 'state', 'zip_code', 'country'].map((k) =>
    `<tr><td class="k">${k}</td><td class="v">${esc(a[k]) || '&mdash;'}</td></tr>`).join('');
  $('card-address').innerHTML = `<table>${rows}</table>${stageMeta(stage)}`;
}

function renderName(stage) {
  if (!stage) return;
  if (!stage.ok) {
    $('card-name').innerHTML = `<div class="error">${esc(stage.error)}</div>${stageMeta(stage)}`;
    return;
  }
  const name = stage.value ? `<div class="big">${esc(stage.value)}</div>`
    : '<span class="empty">(empty)</span>';
  $('card-name').innerHTML = name + stageMeta(stage);
}

function renderType(stage) {
  if (!stage) return;
  if (stage.skipped) {
    $('card-type').innerHTML = `<span class="tag none">skipped</span><div class="meta">${esc(stage.error)}</div>`;
    return;
  }
  if (!stage.ok) {
    $('card-type').innerHTML = `<div class="error">${esc(stage.error)}</div>${stageMeta(stage)}`;
    return;
  }
  const label = stage.value || 'invalid';
  const cls = ['company', 'person'].includes(label) ? label : 'none';
  $('card-type').innerHTML = `<span class="tag ${cls}">${esc(label)}</span>${stageMeta(stage)}`;
}

function renderRaw(stages, totalMs) {
  const parts = [];
  for (const [name, stage] of stages) {
    if (!stage || stage.skipped) continue;
    const lat = stage.latency_ms != null ? ` <span class="lat">&middot; ${stage.latency_ms} ms</span>` : '';
    const body = stage.ok ? `<pre>${esc(stage.raw)}</pre>`
      : `<div class="error">${esc(stage.error)}</div>`;
    parts.push(`<div class="rawstage"><span class="name">${esc(name)}${lat}</span>${body}</div>`);
  }
  if (totalMs != null) parts.push(`<div class="meta">pipeline total: ${totalMs} ms</div>`);
  $('card-raw').innerHTML = parts.length ? parts.join('') : '<span class="empty">No result yet.</span>';
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data && data.detail ? JSON.stringify(data.detail) : `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

const failedStage = (msg) => ({ok: false, error: msg});

async function run(action) {
  const text = $('input').value.trim();
  if (!text) { $('input').focus(); return; }
  const buttons = document.querySelectorAll('button');
  buttons.forEach((b) => b.disabled = true);
  try {
    if (action === 'pipeline') {
      const r = await post('/api/pipeline', {text});
      renderAddress(r.address); renderName(r.entity_name); renderType(r.entity_type);
      renderRaw([['address', r.address], ['entity name', r.entity_name],
                 ['entity type', r.entity_type]], r.total_latency_ms);
    } else if (action === 'address') {
      const r = await post('/api/address', {text});
      renderAddress(r); renderRaw([['address', r]], null);
    } else if (action === 'entity-name') {
      const r = await post('/api/entity-name', {text});
      renderName(r); renderRaw([['entity name', r]], null);
    } else if (action === 'entity-type') {
      const r = await post('/api/entity-type', {name: text});
      renderType(r); renderRaw([['entity type', r]], null);
    }
  } catch (e) {
    const stage = failedStage(e.message);
    if (action === 'pipeline' || action === 'address') renderAddress(stage);
    if (action === 'pipeline' || action === 'entity-name') renderName(stage);
    if (action === 'pipeline' || action === 'entity-type') renderType(stage);
    renderRaw([[action, stage]], null);
  } finally {
    buttons.forEach((b) => b.disabled = false);
  }
}

document.querySelectorAll('button[data-action]').forEach((b) =>
  b.addEventListener('click', () => run(b.dataset.action)));
$('input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run('pipeline');
});

async function checkHealth() {
  const el = $('health');
  el.className = 'down';
  $('health-text').textContent = 'vLLM down';
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    if (h.vllm_up) {
      el.className = 'up';
      $('health-text').textContent = 'vLLM up';
    }
  } catch {
    el.className = 'down';
    $('health-text').textContent = 'vLLM down';
  }
}
checkHealth();
setInterval(checkHealth, 30000);
