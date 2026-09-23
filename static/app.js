/**
 * Python Code Studio - Client Application
 * Communicates with backend.py via standard JSON REST endpoints.
 */

// DOM Elements
const chatEl = document.getElementById('chat');
const emptyState = document.getElementById('empty-state');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const statusDetail = document.getElementById('status-detail');
const statVerifiedFn = document.getElementById('stat-verified-fn');
const statVerifiedProj = document.getElementById('stat-verified-proj');
const clearBtn = document.getElementById('clear-btn');

let isBusy = false;

// ==========================================================================
// Telemetry & Status
// ==========================================================================
async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    statusDot.className = 'pulse-dot ok';
    const paramCount = data.n_params ? `${(data.n_params / 1e6).toFixed(1)}M` : 'Active';
    const dev = (data.device || 'CPU').toUpperCase();
    statusText.textContent = `${paramCount} params (${dev})`;

    if (statVerifiedFn) statVerifiedFn.textContent = data.n_verified_functions ?? '—';
    if (statVerifiedProj) statVerifiedProj.textContent = data.n_verified_projects ?? '—';

    if (statusDetail && data.model_path) {
      const parts = data.model_path.replace(/\\/g, '/').split('/');
      const shortPath = parts.slice(-2).join('/');
      statusDetail.textContent = shortPath;
    }
  } catch (e) {
    statusDot.className = 'pulse-dot error';
    statusText.textContent = 'Backend offline';
    if (statusDetail) statusDetail.textContent = 'Could not establish connection';
  }
}
loadStatus();

// ==========================================================================
// Conversation History
// ==========================================================================
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) return;
    const history = await res.json();
    if (Array.isArray(history) && history.length > 0) {
      if (emptyState) emptyState.style.display = 'none';
      for (const entry of history) {
        if (entry.role === 'user') {
          renderUserMessage(entry.content, false);
        } else if (entry.role === 'assistant' && entry.content) {
          renderAssistantMessage(entry.content, false);
        }
      }
      scrollToBottom();
    }
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}
loadHistory();

// Clear history action
clearBtn?.addEventListener('click', async () => {
  try {
    await fetch('/api/clear_history', { method: 'POST' });
    chatEl.innerHTML = '';
    if (emptyState) {
      chatEl.appendChild(emptyState);
      emptyState.style.display = 'flex';
    }
  } catch (e) {
    console.error('Failed to clear history:', e);
  }
});

// Prompt suggestion cards
document.querySelectorAll('.prompt-card').forEach(card => {
  card.addEventListener('click', () => {
    if (isBusy) return;
    const prompt = card.dataset.prompt;
    if (prompt) {
      inputEl.value = prompt;
      sendMessage();
    }
  });
});

// ==========================================================================
// Composer Interactions
// ==========================================================================
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + 'px';
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatEl.scrollTop = chatEl.scrollHeight;
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

// ==========================================================================
// Message Rendering
// ==========================================================================
function renderUserMessage(text, animate = true) {
  if (emptyState) emptyState.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'msg-row msg-user';
  if (!animate) row.style.animation = 'none';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);

  chatEl.appendChild(row);
  return row;
}

/**
 * Builds a clean code card with syntax highlighting, copy, and download actions.
 */
function buildCodeCard(filename, code, lang = 'python') {
  const card = document.createElement('div');
  card.className = 'code-card';

  const header = document.createElement('div');
  header.className = 'code-card-header';
  header.innerHTML = `
    <div class="file-info">
      <svg class="file-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
      </svg>
      <span>${escapeHtml(filename || 'snippet.py')}</span>
    </div>
    <div class="code-actions">
      <button class="code-btn copy-btn" title="Copy code">
        <svg class="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span class="btn-text">Copy</span>
      </button>
      <button class="code-btn dl-btn" title="Download file">
        <svg class="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        <span class="btn-text">Download</span>
      </button>
    </div>
  `;
  card.appendChild(header);

  const pre = document.createElement('pre');
  const codeEl = document.createElement('code');
  codeEl.className = `language-${lang}`;
  codeEl.textContent = code;
  pre.appendChild(codeEl);
  card.appendChild(pre);

  // Copy handler with visual feedback
  const copyBtn = card.querySelector('.copy-btn');
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code);
      copyBtn.classList.add('copied');
      copyBtn.querySelector('.btn-text').textContent = 'Copied!';
      setTimeout(() => {
        copyBtn.classList.remove('copied');
        copyBtn.querySelector('.btn-text').textContent = 'Copy';
      }, 1500);
    } catch (err) {
      console.error('Failed to copy to clipboard', err);
    }
  });

  // Download handler
  const dlBtn = card.querySelector('.dl-btn');
  dlBtn.addEventListener('click', () => {
    const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'snippet.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Trigger syntax highlighting
  if (window.hljs) {
    hljs.highlightElement(codeEl);
  }

  return card;
}

/**
 * Renders assistant response: handles function vs project modes,
 * status badges, similarity ratings, warnings, and fallback banners.
 */
function renderAssistantMessage(result, animate = true) {
  if (emptyState) emptyState.style.display = 'none';

  const row = document.createElement('div');
  row.className = 'msg-row msg-assistant';
  if (!animate) row.style.animation = 'none';

  // 1. Optional note banner (e.g. fallback to single snippet, or time budget notice)
  if (result.note) {
    const note = document.createElement('div');
    note.className = 'banner-box banner-info';
    note.innerHTML = `
      <svg class="banner-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>
      <span>${escapeHtml(result.note)}</span>
    `;
    row.appendChild(note);
  }

  // 2. Status Badge Row (Verified example vs Generated, Similarity %)
  const badgeRow = document.createElement('div');
  badgeRow.className = 'badge-row';

  const isVerified = result.source === 'verified';
  const badge = document.createElement('div');
  badge.className = `status-pill ${isVerified ? 'pill-verified' : 'pill-generated'}`;

  const dot = document.createElement('span');
  dot.className = 'pill-dot';
  badge.appendChild(dot);

  const labelSpan = document.createElement('span');
  labelSpan.textContent = isVerified ? 'Verified example' : 'Generated model output';
  badge.appendChild(labelSpan);

  if (result.similarity !== null && result.similarity !== undefined) {
    const simSpan = document.createElement('span');
    simSpan.className = 'match-similarity';
    simSpan.textContent = `· ${(result.similarity * 100).toFixed(0)}% match`;
    badge.appendChild(simSpan);
  }

  badgeRow.appendChild(badge);
  row.appendChild(badgeRow);

  // 3. Single Function Mode vs Multi-File Project Mode
  if (result.kind === 'function') {
    row.appendChild(buildCodeCard('snippet.py', result.code, 'python'));

    if (result.warning) {
      const warn = document.createElement('div');
      warn.className = 'banner-box banner-warning';
      warn.innerHTML = `
        <svg class="banner-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
          <line x1="12" y1="9" x2="12" y2="13"></line>
          <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
        <span>${escapeHtml(result.warning)}</span>
      `;
      row.appendChild(warn);
    }
  } else {
    // Project Mode
    const projectCard = document.createElement('div');
    projectCard.className = 'project-card';

    // File structure visual box
    if (result.structure) {
      const treeBox = document.createElement('div');
      treeBox.className = 'project-tree-box';
      const treeHeader = document.createElement('div');
      treeHeader.className = 'tree-header';
      treeHeader.textContent = 'Project Directory Architecture';
      treeBox.appendChild(treeHeader);

      const treeContent = document.createElement('div');
      treeContent.textContent = result.structure;
      treeBox.appendChild(treeContent);
      projectCard.appendChild(treeBox);
    }

    // File tabs and active viewer
    const filenames = Object.keys(result.files || {});
    if (filenames.length > 0) {
      const browser = document.createElement('div');
      browser.className = 'project-browser';

      const tabStrip = document.createElement('div');
      tabStrip.className = 'file-tab-bar';

      const codeArea = document.createElement('div');

      function switchTab(targetName) {
        codeArea.innerHTML = '';
        const lang = targetName.endsWith('.py') ? 'python' : 'plaintext';
        codeArea.appendChild(buildCodeCard(targetName, result.files[targetName], lang));

        tabStrip.querySelectorAll('.file-tab-btn').forEach(btn => {
          btn.classList.toggle('active', btn.dataset.file === targetName);
        });
      }

      filenames.forEach(fname => {
        const tabBtn = document.createElement('button');
        tabBtn.className = 'file-tab-btn';
        tabBtn.dataset.file = fname;
        tabBtn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
          <span>${escapeHtml(fname)}</span>
        `;
        tabBtn.addEventListener('click', () => switchTab(fname));
        tabStrip.appendChild(tabBtn);
      });

      browser.appendChild(tabStrip);
      browser.appendChild(codeArea);
      projectCard.appendChild(browser);

      // Select initial tab
      switchTab(filenames[0]);

      // Download project zip button
      const zipBtn = document.createElement('button');
      zipBtn.className = 'zip-download-btn';
      zipBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="21 8 21 21 3 21 3 8"></polyline>
          <rect x="1" y="3" width="22" height="5"></rect>
          <line x1="10" y1="12" x2="14" y2="12"></line>
        </svg>
        <span>Download Project (.zip)</span>
      `;
      zipBtn.addEventListener('click', async () => {
        try {
          zipBtn.disabled = true;
          const originalText = zipBtn.innerHTML;
          zipBtn.innerHTML = `<span>Packing archive...</span>`;

          const res = await fetch('/api/download_project', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: result.files }),
          });

          if (!res.ok) throw new Error('Download failed');
          const blob = await res.blob();
          const downloadUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = downloadUrl;
          a.download = 'project.zip';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(downloadUrl);

          zipBtn.innerHTML = originalText;
          zipBtn.disabled = false;
        } catch (e) {
          console.error('ZIP download error:', e);
          zipBtn.textContent = 'Download error';
          setTimeout(() => {
            zipBtn.disabled = false;
            zipBtn.innerHTML = `<span>Download Project (.zip)</span>`;
          }, 2000);
        }
      });
      projectCard.appendChild(zipBtn);
    }

    row.appendChild(projectCard);
  }

  chatEl.appendChild(row);
  return row;
}

/**
 * Renders calm, understated thinking state (pulsing dots rather than spinning neon ring).
 */
function renderThinking() {
  const row = document.createElement('div');
  row.className = 'msg-row msg-assistant';
  row.id = 'thinking-row';
  row.innerHTML = `
    <div class="thinking-box">
      <div class="thinking-dots">
        <span></span><span></span><span></span>
      </div>
      <span class="thinking-label">Synthesizing response...</span>
    </div>
  `;
  chatEl.appendChild(row);
  scrollToBottom();
}

/**
 * Handles message submission to /api/chat.
 */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isBusy) return;

  isBusy = true;
  sendBtn.disabled = true;

  renderUserMessage(text, true);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  scrollToBottom();
  renderThinking();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });

    const result = await res.json();
    document.getElementById('thinking-row')?.remove();

    if (result.error) {
      const errRow = document.createElement('div');
      errRow.className = 'msg-row msg-assistant';
      errRow.innerHTML = `
        <div class="banner-box banner-warning">
          <svg class="banner-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span>${escapeHtml(result.error)}</span>
        </div>
      `;
      chatEl.appendChild(errRow);
    } else {
      renderAssistantMessage(result, true);
    }
  } catch (err) {
    document.getElementById('thinking-row')?.remove();
    const errRow = document.createElement('div');
    errRow.className = 'msg-row msg-assistant';
    errRow.innerHTML = `
      <div class="banner-box banner-warning">
        <svg class="banner-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <span>Connection failure: ${escapeHtml(String(err.message || err))}</span>
      </div>
    `;
    chatEl.appendChild(errRow);
  }

  scrollToBottom();
  isBusy = false;
  sendBtn.disabled = false;
  inputEl.focus();
}
