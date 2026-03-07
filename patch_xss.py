import sys

def apply_h_escapes():
    with open('server.py', 'r', encoding='utf-8') as f:
        new_content = f.read()

    replacements = [
        (
            """function renderImpactSidebar(node, impact) {
  const br = impact.blast_radius || {};
  const meta = impact.target_meta || {};
  const sev = impact.severity || 'low';""",
            """function renderImpactSidebar(node, impact) {
  const br = impact.blast_radius || {};
  const meta = impact.target_meta || {};
  const sev = _h(impact.severity || 'low');"""
        ),
        (
            """let html = `
    <div class="meta-row">
      <span class="meta-pill">${meta.type || node.type}</span>""",
            """let html = `
    <div class="meta-row">
      <span class="meta-pill">${_h(meta.type || node.type)}</span>"""
        ),
        (
            """if (meta.file) {
    const fname = meta.file.split(/[\\/\\\\]/).pop();
    html += `<div style="color:#8b949e;font-size:10px;margin-top:4px">${fname}:${meta.line || 0}</div>`;
  }
  if (meta.signature) {
    html += `<div style="color:#79c0ff;font-size:11px;margin-top:6px;font-family:'SF Mono',monospace">${node.name}${meta.signature}</div>`;
  }
  if (meta.docstring) {
    html += `<div style="color:#6e7681;font-size:10px;margin-top:5px;line-height:1.5">${meta.docstring}</div>`;
  }""",
            """if (meta.file) {
    const fname = meta.file.split(/[\\/\\\\]/).pop();
    html += `<div style="color:#8b949e;font-size:10px;margin-top:4px">${_h(fname)}:${_h(meta.line || 0)}</div>`;
  }
  if (meta.signature) {
    html += `<div style="color:#79c0ff;font-size:11px;margin-top:6px;font-family:'SF Mono',monospace">${_h(node.name)}${_h(meta.signature)}</div>`;
  }
  if (meta.docstring) {
    html += `<div style="color:#6e7681;font-size:10px;margin-top:5px;line-height:1.5">${_h(meta.docstring)}</div>`;
  }"""
        ),
        (
            """html += `<div style="color:#8b949e;font-size:10px;margin-top:10px">${impact.llm_context_hint || ''}</div>`;""",
            """html += `<div style="color:#8b949e;font-size:10px;margin-top:10px">${_h(impact.llm_context_hint || '')}</div>`;"""
        ),
        (
            """html += `<div class="impact-section"><h3>Critical path</h3>
      <div style="font-size:10px;color:#8b949e;word-break:break-all">
        ${impact.critical_path.map(n => `<span style="color:#79c0ff">${n.split('.').pop()}</span>`).join(' → ')}
      </div></div>`;""",
            """html += `<div class="impact-section"><h3>Critical path</h3>
      <div style="font-size:10px;color:#8b949e;word-break:break-all">
        ${impact.critical_path.map(n => `<span style="color:#79c0ff">${_h(n.split('.').pop())}</span>`).join(' → ')}
      </div></div>`;"""
        ),
        (
            """function renderImpactNode(e) {
  const sev = e.severity || 'low';
  return `<div class="impact-node" onclick="prefillConsequenceContext('${e.node}')">
    <div class="impact-node-title">
      <span class="severity-dot bg-${sev}"></span>
      <span class="node-name">${e.name}</span>
      <span class="edge-type">${e.edge_type} (depth ${e.depth})</span>
    </div>
    <div class="impact-reason">${e.reason}</div>
  </div>`;
}""",
            """function renderImpactNode(e) {
  const sev = e.severity || 'low';
  return `<div class="impact-node" onclick="prefillConsequenceContext('${_h(e.node)}')">
    <div class="impact-node-title">
      <span class="severity-dot bg-${_h(sev)}"></span>
      <span class="node-name">${_h(e.name)}</span>
      <span class="edge-type">${_h(e.edge_type)} (depth ${_h(e.depth)})</span>
    </div>
    <div class="impact-reason">${_h(e.reason)}</div>
  </div>`;
}"""
        ),
        (
            """html = `
        <div class="summary-box">
          ${lead.summary}
        </div>""",
            """html = `
        <div class="summary-box">
          ${_h(lead.summary)}
        </div>"""
        ),
        (
            """lead.steps.forEach(s => {
        html += `
          <div class="plan-step" onclick="selectNodeById('${s.node}')">
            <div class="step-num">${s.step}</div>
            <div class="step-content">
              <div class="step-target">${s.name}</div>
              <div class="step-action">${s.action}</div>
            </div>
          </div>
        `;
      });""",
            """lead.steps.forEach(s => {
        html += `
          <div class="plan-step" onclick="selectNodeById('${_h(s.node)}')">
            <div class="step-num">${_h(s.step)}</div>
            <div class="step-content">
              <div class="step-target">${_h(s.name)}</div>
              <div class="step-action">${_h(s.action)}</div>
            </div>
          </div>
        `;
      });"""
        ),
        (
            """lead.ordered_steps.forEach(s => {
        html += `
          <div class="plan-step" onclick="selectNodeById('${s.node}')">
            <div class="step-num">${s.step}</div>
            <div class="step-content">
              <div class="step-target">${s.name}</div>
              <div class="step-action">${s.action}</div>
            </div>
          </div>
        `;
      });""",
            """lead.ordered_steps.forEach(s => {
        html += `
          <div class="plan-step" onclick="selectNodeById('${_h(s.node)}')">
            <div class="step-num">${_h(s.step)}</div>
            <div class="step-content">
              <div class="step-target">${_h(s.name)}</div>
              <div class="step-action">${_h(s.action)}</div>
            </div>
          </div>
        `;
      });"""
        ),
        (
            """html = `
        <div class="decision-card">
          <div class="decision-header">
            <span class="decision-intent">${intentLabel}</span>
            <span class="decision-change-type">${changeLabel}</span>
          </div>
          <div class="decision-recommendation">${recommendation}</div>
          <div class="decision-options">
      `;
      lead.options.forEach(opt => {
        const p = Math.min(100, Math.round((opt.downstream_count / Math.max(1, opt.downstream_count * 2)) * 100));
        html += `
          <div class="decision-option" onclick="selectNodeById('${opt.node}')">
            <div class="opt-header">
              <span class="opt-name">${opt.name}</span>
              <span class="opt-cost">cascades to ${opt.downstream_count}</span>
            </div>
            <div class="opt-bar"><div class="opt-fill" style="width:${p}%"></div></div>
          </div>`;
      });""",
            """html = `
        <div class="decision-card">
          <div class="decision-header">
            <span class="decision-intent">${_h(intentLabel)}</span>
            <span class="decision-change-type">${_h(changeLabel)}</span>
          </div>
          <div class="decision-recommendation">${recommendation}</div>
          <div class="decision-options">
      `;
      lead.options.forEach(opt => {
        const p = Math.min(100, Math.round((opt.downstream_count / Math.max(1, opt.downstream_count * 2)) * 100));
        html += `
          <div class="decision-option" onclick="selectNodeById('${_h(opt.node)}')">
            <div class="opt-header">
              <span class="opt-name">${_h(opt.name)}</span>
              <span class="opt-cost">cascades to ${_h(opt.downstream_count)}</span>
            </div>
            <div class="opt-bar"><div class="opt-fill" style="width:${p}%"></div></div>
          </div>`;
      });"""
        ),
        (
            """html += `<div class="scope-row" onclick="selectNodeById('${n.node}')">
        <span class="scope-name">${n.name}</span>
      </div>`;""",
            """html += `<div class="scope-row" onclick="selectNodeById('${_h(n.node)}')">
        <span class="scope-name">${_h(n.name)}</span>
      </div>`;"""
        ),
        (
            """function renderCqNode(n, tier) {
  const dotColor = tier === 1 ? '#f85149' : tier === 2 ? '#e3b341' : '#6e7681';
  const pillLabel = tier === 1 ? 'breaks' : tier === 2 ? 'review' : 'in scope';

  const hookBadge = n.is_hook
    ? `<span class="detail-badge" style="color:#f0883e;border-color:#f0883e33;background:#f0883e11">hook</span>` : '';
  const shapesBadge = Object.keys(n.shapes || {}).length
    ? `<span class="detail-badge" style="color:#d2a8ff;border-color:#d2a8ff33;background:#d2a8ff11">tensor_contract</span>` : '';

  const fname = n.file ? n.file.split(/[\\/\\\\]/).pop() : '';
  const loc = fname ? `<span class="detail-loc">${fname}${n.line ? ':' + n.line : ''}</span>` : '';

  const edgeTypes = n.edge_types && n.edge_types.length > 0
    ? `<span class="detail-badge">${n.edge_types.slice(0, 2).join(', ')}</span>` : '';

  const viaStr = n.via && n.via.length ? ` <span style="color:#8b949e">via</span> ${n.via.join(', ')}` : '';
  const sharedBadge = n.intersection_count > 1 ? ` <span class="detail-badge" style="color:#7ee787">shared (${n.intersection_count})</span>` : '';

  return `
    <div class="cq-node tier-${tier}" onclick="selectNodeById('${n.node}')">
      <div class="cq-node-top">
        <span class="cq-node-title">
          <span style="color:${dotColor};font-size:10px;margin-right:2px">●</span>
          ${n.name}
          ${sharedBadge}
        </span>
        <span class="cq-node-pill" style="color:${dotColor};border-color:${dotColor}33">${pillLabel}</span>
      </div>
      <div class="cq-node-detail">
        ${loc} ${hookBadge} ${shapesBadge} ${edgeTypes} ${viaStr}
      </div>
      <div class="cq-consequence-msg">${n.consequence}</div>
    </div>
  `;
}""",
            """function renderCqNode(n, tier) {
  const dotColor = tier === 1 ? '#f85149' : tier === 2 ? '#e3b341' : '#6e7681';
  const pillLabel = tier === 1 ? 'breaks' : tier === 2 ? 'review' : 'in scope';

  const hookBadge = n.is_hook
    ? `<span class="detail-badge" style="color:#f0883e;border-color:#f0883e33;background:#f0883e11">hook</span>` : '';
  const shapesBadge = Object.keys(n.shapes || {}).length
    ? `<span class="detail-badge" style="color:#d2a8ff;border-color:#d2a8ff33;background:#d2a8ff11">tensor_contract</span>` : '';

  const fname = n.file ? n.file.split(/[\\/\\\\]/).pop() : '';
  const loc = fname ? `<span class="detail-loc">${_h(fname)}${n.line ? ':' + _h(n.line) : ''}</span>` : '';

  const edgeTypes = n.edge_types && n.edge_types.length > 0
    ? `<span class="detail-badge">${_h(n.edge_types.slice(0, 2).join(', '))}</span>` : '';

  const viaStr = n.via && n.via.length ? ` <span style="color:#8b949e">via</span> ${_h(n.via.join(', '))}` : '';
  const sharedBadge = n.intersection_count > 1 ? ` <span class="detail-badge" style="color:#7ee787">shared (${_h(n.intersection_count)})</span>` : '';

  return `
    <div class="cq-node tier-${tier}" onclick="selectNodeById('${_h(n.node)}')">
      <div class="cq-node-top">
        <span class="cq-node-title">
          <span style="color:${dotColor};font-size:10px;margin-right:2px">●</span>
          ${_h(n.name)}
          ${sharedBadge}
        </span>
        <span class="cq-node-pill" style="color:${dotColor};border-color:${dotColor}33">${_h(pillLabel)}</span>
      </div>
      <div class="cq-node-detail">
        ${loc} ${hookBadge} ${shapesBadge} ${edgeTypes} ${viaStr}
      </div>
      <div class="cq-consequence-msg">${n.consequence}</div>
    </div>
  `;
}"""
        ),
        (
            """html += `<div class="arity-card" onclick="selectNodeById('${w.node}')">
        <div class="arity-node">${w.name}</div>
        <div class="arity-loc">${loc} · <span style="color:#8b949e">expects ${w.consumer_arity} ↔ source returns ${w.source_arity}</span></div>
        <div class="arity-msg">${msg}</div>
      </div>`;""",
            """html += `<div class="arity-card" onclick="selectNodeById('${_h(w.node)}')">
        <div class="arity-node">${_h(w.name)}</div>
        <div class="arity-loc">${loc} · <span style="color:#8b949e">expects ${_h(w.consumer_arity)} ↔ source returns ${_h(w.source_arity)}</span></div>
        <div class="arity-msg">${msg}</div>
      </div>`;"""
        ),
        (
            """data.results.forEach(r => {
        html += `<div class="res" onclick="selectNodeById('${r.id}')">
          <div class="res-title">${r.name}<span class="res-type">${r.type}</span></div>
          <div class="res-path">${r.file}</div>
        </div>`;
      });""",
            """data.results.forEach(r => {
        html += `<div class="res" onclick="selectNodeById('${_h(r.id)}')">
          <div class="res-title">${_h(r.name)}<span class="res-type">${_h(r.type)}</span></div>
          <div class="res-path">${_h(r.file)}</div>
        </div>`;
      });"""
        ),
        (
            """html += `<div class="impact-node" onclick="selectNodeById('${imp.target || ''}')">
          <div class="impact-node-title">
            <span class="severity-dot sev-${imp.severity}"></span>
            <span class="node-name">${imp.function}</span>
          </div>
          <div class="impact-reason">Impact: ${imp.severity} (${imp.downstream_count} downstream files)</div>
        </div>`;""",
            """html += `<div class="impact-node" onclick="selectNodeById('${_h(imp.target) || ''}')">
          <div class="impact-node-title">
            <span class="severity-dot sev-${_h(imp.severity)}"></span>
            <span class="node-name">${_h(imp.function)}</span>
          </div>
          <div class="impact-reason">Impact: ${_h(imp.severity)} (${_h(imp.downstream_count)} downstream files)</div>
        </div>`;"""
        )
    ]

    for old, new in replacements:
        new_content = new_content.replace(old, new)
        
    with open('server.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == '__main__':
    apply_h_escapes()
    print("Done")
