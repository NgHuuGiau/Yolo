/**
 * OncoVision AI Workstation Client Application
 */

const CANCER_TARGETS = window.CANCER_TARGETS || [];
let conversations = [];
let activeConvId = null;
let activeConvIndex = 0;
let pendingAttachments = [];
let currentTheme = localStorage.getItem('theme') || 'system';
let currentLang = localStorage.getItem('lang') || 'vi';
let isRecording = false;
let recognition = null;
let isAnalyzing = false;

const TR = {
  vi: {
    new_chat: 'Ca chẩn đoán mới',
    search: 'Tìm ca bệnh/bệnh nhân...',
    history: 'Hồ Sơ Ca Lâm Sàng',
    settings: 'Cấu hình hệ thống',
    app_name: 'OncoVision AI™',
    greeting_title: 'Hệ Thống Trợ Lý Chẩn Đoán',
    greeting_text: 'Xin chào! Hãy tải phim chụp lên bảng điều khiển trung tâm hoặc đặt câu hỏi y khoa dưới đây để nhận hỗ trợ sàng lọc tự động.',
    input_placeholder: 'Nhập câu hỏi lâm sàng tại đây...',
    empty_title: 'Tư Vấn & Đọc Kết Quả',
    empty_subtitle: 'Gửi ảnh chụp hoặc văn bản mô tả bệnh lý để nhận phân tích chi tiết tự động từ mô hình chuyên biệt.',
    disclaimer: 'Kết quả phân tích từ AI chỉ mang tính tham khảo hỗ trợ, vui lòng xác chẩn với bác sĩ chuyên khoa lâm sàng.',
    attach: 'Đính kèm',
    camera: 'Mở camera',
    choose_image: 'Chọn ảnh chụp',
    medical_target: 'Nhóm bệnh học',
    medical_modality: 'Loại ảnh (Modality)',
    light: 'Clinic Mode (Sáng)',
    dark: 'PACS Mode (Tối)',
    system: 'Hệ thống',
    today: 'Hôm nay',
    medical_analyzing: 'Đang phân tích tổn thương y khoa...',
    medical_pending: 'Đang chạy pipeline sàng lọc y dược...',
    system_reply_text: 'Tôi đã nhận nội dung và sẽ xử lý trong ngữ cảnh ca bệnh này.',
  },
  en: {
    new_chat: 'New Diagnostic Case',
    search: 'Search case/patient...',
    history: 'Clinical Case Logs',
    settings: 'System Settings',
    app_name: 'OncoVision AI™',
    greeting_title: 'Diagnostic Assistant System',
    greeting_text: 'Welcome! Upload a medical scan to the center view or write a clinical query below for automated analysis support.',
    input_placeholder: 'Enter clinical query here...',
    empty_title: 'AI Clinical Companion',
    empty_subtitle: 'Submit medical imaging or case details to obtain specialized automated classification and report.',
    disclaimer: 'AI results are for diagnostic reference only. Please confirm with clinical specialists.',
    attach: 'Attach',
    camera: 'Open camera',
    choose_image: 'Choose image',
    medical_target: 'Cancer Targets',
    medical_modality: 'Imaging Modality',
    light: 'Clinic Mode (Light)',
    dark: 'PACS Mode (Dark)',
    system: 'System Default',
    today: 'Today',
    medical_analyzing: 'Analyzing medical imaging lesion...',
    medical_pending: 'Medical analysis pipeline is running...',
    system_reply_text: 'I received the content and will process it locally in this case context.',
  }
};

function t(key) {
  return (TR[currentLang] || TR.vi)[key] || key;
}

function applyLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `language=${encodeURIComponent(lang)}&theme=${encodeURIComponent(currentTheme)}`
  }).catch(() => {});
  translateUI();
}

function translateUI() {
  const newChatSpan = document.querySelector('.new-chat-btn span');
  if (newChatSpan) newChatSpan.textContent = t('new_chat');
  const searchInput = document.getElementById('searchInput');
  if (searchInput) searchInput.placeholder = t('search');
  const histTitle = document.querySelector('.history-title');
  if (histTitle) histTitle.textContent = t('history');
  const settingsBtnSpan = document.querySelector('.settings-btn span');
  if (settingsBtnSpan) settingsBtnSpan.textContent = t('settings');
  const msgInput = document.getElementById('messageInput');
  if (msgInput) msgInput.placeholder = t('input_placeholder');
  const greetingH3 = document.querySelector('.greeting-card h3');
  if (greetingH3) greetingH3.textContent = t('greeting_title');
  const greetingP = document.querySelector('.greeting-card p');
  if (greetingP) greetingP.textContent = t('greeting_text');
  const emptyH3 = document.querySelector('.empty-state h3');
  if (emptyH3) emptyH3.textContent = t('empty_title');
  const emptyP = document.querySelector('.empty-state p');
  if (emptyP) emptyP.textContent = t('empty_subtitle');
  const disclaimer = document.querySelector('.disclaimer');
  if (disclaimer) disclaimer.textContent = t('disclaimer');
  const labels = document.querySelectorAll('.control-group label');
  if (labels.length >= 2) {
    labels[0].textContent = t('medical_target');
    labels[1].textContent = t('medical_modality');
  }
}

function setTheme(mode) {
  currentTheme = mode;
  localStorage.setItem('theme', mode);
  applyThemeMode();
  const lightBtn = document.getElementById('lightBtn');
  const darkBtn = document.getElementById('darkBtn');
  const sysBtn = document.getElementById('systemBtn');
  if (lightBtn) lightBtn.classList.toggle('active', mode === 'light');
  if (darkBtn) darkBtn.classList.toggle('active', mode === 'dark');
  if (sysBtn) sysBtn.classList.toggle('active', mode === 'system');
  const ts = document.getElementById('themeSetting');
  if (ts) ts.value = mode;
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `language=${encodeURIComponent(currentLang)}&theme=${encodeURIComponent(mode)}`
  }).catch(() => {});
}

function applyThemeMode() {
  const mode = currentTheme === 'system'
    ? (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
    : currentTheme;
  document.body.classList.toggle('light', mode === 'light');
}

if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', applyThemeMode);
}

function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  if (sb) sb.classList.toggle('collapsed');
}

function initTargets() {
  const s = document.getElementById('targetSelect');
  if (!s || !CANCER_TARGETS.length) return;
  s.innerHTML = '';
  CANCER_TARGETS.forEach(target => {
    const o = document.createElement('option');
    o.value = target.key;
    o.textContent = target.label;
    s.appendChild(o);
  });
  updateModalities();
  s.onchange = () => {
    const found = CANCER_TARGETS.find(x => x.key === s.value);
    if (found) {
      updateModalityList(found.modalities);
      const th = document.getElementById('targetHint');
      if (th) th.textContent = 'Ảnh thường dùng: ' + found.modalities.join(', ');
    }
  };
  s.onchange();
}

function updateModalities() {
  const first = CANCER_TARGETS[0];
  if (first) updateModalityList(first.modalities);
  const th = document.getElementById('targetHint');
  if (th) th.textContent = 'Ảnh thường dùng: ' + (first ? first.modalities.join(', ') : '');
}

function updateModalityList(modalities) {
  const s = document.getElementById('modalitySelect');
  if (!s) return;
  s.innerHTML = '';
  (modalities || []).forEach(m => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = m;
    s.appendChild(o);
  });
  s.onchange = () => {
    const mh = document.getElementById('modalityHint');
    if (mh) mh.textContent = 'Modality đã chọn: ' + s.value;
  };
  s.onchange();
}

function applyThemeSetting(v) {
  setTheme(v);
}

function openSettings() {
  const m = document.getElementById('settingsModal');
  if (m) m.classList.add('active');
}

function closeSettings() {
  const m = document.getElementById('settingsModal');
  if (m) m.classList.remove('active');
}

async function loadConversations() {
  try {
    const r = await fetch('/api/conversations');
    const d = await r.json();
    conversations = d.conversations || [];
  } catch (e) {
    conversations = [];
  }
  if (!conversations.length) {
    conversations = [{ id: null, title: t('new_chat'), subtitle: t('today'), messages: [] }];
  }
  renderHistory();
  if (activeConvId === null && conversations[0]) {
    selectConv(0);
  }
}

function renderHistory() {
  const searchEl = document.getElementById('searchInput');
  const q = (searchEl ? searchEl.value : '').toLowerCase().trim();
  const p = document.getElementById('historyPanel');
  if (!p) return;
  p.innerHTML = '';
  conversations.forEach((c, i) => {
    if (q && !c.title.toLowerCase().includes(q) && !(c.subtitle || '').toLowerCase().includes(q)) return;
    const d = document.createElement('div');
    d.className = 'history-item' + (i === activeConvIndex ? ' active' : '');
    d.innerHTML = `
      <div class="history-item-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <div class="history-item-text">
        <div class="history-item-title">${escHtml(c.title || t('new_chat'))}</div>
        <div class="history-item-subtitle">${escHtml(c.subtitle || '')}</div>
      </div>
    `;
    d.onclick = () => selectConv(i);
    d.oncontextmenu = e => {
      e.preventDefault();
      if (confirm('Xóa trò chuyện này?')) deleteConv(i);
    };
    p.appendChild(d);
  });
}

function filterHistory() {
  renderHistory();
}

async function newChat() {
  try {
    const r = await fetch('/api/conversations', { method: 'POST' });
    const d = await r.json();
    conversations.unshift({ id: d.conversation_id, title: t('new_chat'), subtitle: t('today'), messages: [] });
    selectConv(0);
  } catch (e) {
    conversations.unshift({ id: null, title: t('new_chat'), subtitle: t('today'), messages: [] });
    selectConv(0);
  }
}

async function selectConv(i) {
  if (i < 0 || i >= conversations.length) return;
  activeConvIndex = i;
  const c = conversations[i];
  if (c.id) {
    try {
      const r = await fetch(`/api/conversations/${c.id}`);
      const d = await r.json();
      if (d.ok) {
        c.messages = d.conversation.messages;
      }
    } catch (e) {}
  }
  activeConvId = c.id;
  renderHistory();
  renderMessages();
  syncPACSView();
}

async function deleteConv(i) {
  const c = conversations[i];
  if (c && c.id) {
    try {
      await fetch(`/api/conversations/${c.id}`, { method: 'DELETE' });
    } catch (e) {}
  }
  conversations.splice(i, 1);
  if (!conversations.length) {
    conversations.push({ id: null, title: t('new_chat'), subtitle: t('today'), messages: [] });
  }
  selectConv(Math.min(activeConvIndex, conversations.length - 1));
}

function syncPACSView() {
  const c = conversations[activeConvIndex];
  let lastImage = null;
  let patientCode = "WEB";

  if (c && c.subtitle) {
    patientCode = c.subtitle;
  }

  if (c && c.messages) {
    for (let j = c.messages.length - 1; j >= 0; j--) {
      const msg = c.messages[j];
      if (msg.attachment_path && (msg.attachment_kind === 'image' || msg.attachment_kind === 'camera')) {
        lastImage = msg.attachment_path;
        break;
      }
    }
  }

  const emptyDropzone = document.getElementById('emptyDropzone');
  const pacsImageContainer = document.getElementById('pacsImageContainer');
  const pacsActiveImage = document.getElementById('pacsActiveImage');
  const pacsPatientCodeEl = document.getElementById('pacsPatientCode');
  const pacsDateEl = document.getElementById('pacsDate');

  if (pacsPatientCodeEl) pacsPatientCodeEl.textContent = patientCode;

  if (lastImage) {
    if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(lastImage);
    if (emptyDropzone) emptyDropzone.style.display = 'none';
    if (pacsImageContainer) pacsImageContainer.style.display = 'flex';
    if (pacsDateEl) pacsDateEl.textContent = new Date().toLocaleDateString('vi-VN');
  } else {
    if (pacsActiveImage) pacsActiveImage.src = '';
    if (emptyDropzone) emptyDropzone.style.display = 'flex';
    if (pacsImageContainer) pacsImageContainer.style.display = 'none';
    if (pacsDateEl) pacsDateEl.textContent = '-';
  }
}

function renderMessages() {
  const c = conversations[activeConvIndex];
  const hasMsg = c && c.messages && c.messages.length > 0;
  const greeting = document.getElementById('greetingCard');
  const empty = document.getElementById('emptyState');
  const container = document.getElementById('messagesContainer');
  if (greeting) greeting.style.display = hasMsg ? 'none' : 'block';
  if (empty) empty.style.display = hasMsg ? 'none' : 'flex';
  if (container) container.style.display = hasMsg ? 'block' : 'none';
  if (!hasMsg || !container) return;
  container.innerHTML = '';
  c.messages.forEach(msg => addBubble(msg, false));
  const area = document.getElementById('messagesArea');
  if (area) area.scrollTop = area.scrollHeight;
}

function attachmentUrl(p) {
  if (!p) return '';
  const norm = String(p).replace(/\\/g, '/');
  const i = norm.toLowerCase().lastIndexOf('/output/');
  if (i >= 0) return norm.slice(i);
  return '/' + norm.replace(/^\/+/, '');
}

function addBubble(msg, animate = true) {
  const container = document.getElementById('messagesContainer');
  if (!container) return;
  const row = document.createElement('div');
  row.className = 'message-row ' + (msg.sender || 'assistant');
  if (!animate) row.style.animation = 'none';
  let html = '<div class="bubble">';
  if (msg.attachment_path && msg.attachment_kind) {
    const fname = msg.attachment_path.split('/').pop().split('\\').pop();
    html += `<div class="attachment-name">📎 ${escHtml(fname)}</div>`;
    if (msg.attachment_kind === 'image' || msg.attachment_kind === 'camera') {
      html += `<img src="${attachmentUrl(msg.attachment_path)}" onclick="previewImage(this.src)" loading="lazy">`;
    }
  }
  const txt = msg.text || '';
  let meta = null;
  try {
    meta = JSON.parse(msg.metadata_json || 'null');
  } catch (e) {}
  if (/Cannot read|does not support image/i.test(txt)) {
    html += escHtml('Đã nhận ảnh: ' + (msg.attachment_path || ''));
  } else if (txt.startsWith('Phân tích lỗi:') || txt.startsWith('Error:')) {
    html += `<div class="error-text">${escHtml(txt)}</div>`;
  } else {
    html += escHtml(txt);
  }
  if (meta && meta.medical_case_id) {
    html += buildResultCard(meta);
  }
  html += '</div>';
  row.innerHTML = html;
  container.appendChild(row);
  if (animate) {
    setTimeout(() => {
      const area = document.getElementById('messagesArea');
      if (area) area.scrollTop = area.scrollHeight;
    }, 50);
  }
}

function showAILoadingIndicator() {
  hideAILoadingIndicator();
  const container = document.getElementById('messagesContainer');
  if (!container) return;
  const row = document.createElement('div');
  row.id = 'aiLoadingRow';
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="ai-loading-bubble">
      <div class="ai-loading-dots">
        <span></span><span></span><span></span>
      </div>
      <span>${t('medical_analyzing')}</span>
    </div>
  `;
  container.appendChild(row);
  const area = document.getElementById('messagesArea');
  if (area) area.scrollTop = area.scrollHeight;
}

function hideAILoadingIndicator() {
  const row = document.getElementById('aiLoadingRow');
  if (row && row.parentNode) {
    row.parentNode.removeChild(row);
  }
}

function buildResultCard(m) {
  const risk = (m.risk_level || 'uncertain').toLowerCase();
  const dets = m.detections || [];
  const gradcams = (m.gradcam_overlays || []).filter(Boolean);
  let detRows = dets.slice(0, 5).map(d =>
    `<tr><td>${escHtml(d.label)}</td><td><span class="conf-bar-wrap"><span class="conf-bar-fill" style="width:${Math.round(d.confidence * 100)}%"></span></span>${(d.confidence * 100).toFixed(1)}%</td></tr>`
  ).join('');
  if (!detRows) detRows = '<tr><td>Không ghi nhận vùng nghi ngờ rõ ràng</td></tr>';
  const gradcamHtml = gradcams.length
    ? `<div class="gradcam-strip">${gradcams.map(p => `<img src="${attachmentUrl(p)}" onclick="previewImage(this.src)" title="Grad-CAM heatmap" loading="lazy">`).join('')}</div>`
    : '';
  return `
    <div class="result-card">
      <div class="result-card-head">
        <b>Kết quả phân tích AI</b>
        <span class="risk-badge risk-${risk}">${escHtml(risk)}</span>
      </div>
      <div class="result-stats">
        <div class="result-stat"><b>${dets.length}</b><span>Vùng phát hiện</span></div>
        <div class="result-stat"><b>${((m.average_confidence || 0) * 100).toFixed(0)}%</b><span>Tin cậy TB</span></div>
        <div class="result-stat"><b>${m.suspected_malignant ? 'Có' : 'Không'}</b><span>Nghi ác tính</span></div>
      </div>
      <table class="result-detections">${detRows}</table>
      ${gradcamHtml}
      <div class="result-actions">
        <a class="result-btn" href="/api/cases/${m.medical_case_id}/pdf" download>⬇ Xuất PDF</a>
        <button class="result-btn" onclick="openCompare(${m.medical_case_id})">⇄ So sánh</button>
      </div>
    </div>
  `;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function togglePlusMenu(e) {
  const m = document.getElementById('plusMenu');
  if (!m) return;
  if (m.style.display === 'block') {
    m.style.display = 'none';
    return;
  }
  m.style.display = 'block';
  m.innerHTML = `
    <button onclick="pickImage()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
      </svg>
      ${t('choose_image')}
    </button>
    <button onclick="openCamera()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>
      </svg>
      ${t('camera')}
    </button>
  `;
  const rect = e.currentTarget.getBoundingClientRect();
  m.style.top = (rect.bottom + 4) + 'px';
  m.style.left = rect.left + 'px';
}

document.addEventListener('click', e => {
  const m = document.getElementById('plusMenu');
  if (m && !e.target.closest('.plus-btn') && !e.target.closest('.plus-menu')) {
    m.style.display = 'none';
  }
});

function pickImage() {
  const m = document.getElementById('plusMenu');
  if (m) m.style.display = 'none';
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = 'image/*,.dcm';
  inp.onchange = async () => {
    if (!inp.files[0]) return;
    const f = inp.files[0];
    const fd = new FormData();
    fd.append('file', f);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.ok) {
        addAttachment(d.stored_path, 'image', f.name);
        if (d.detected_target) {
          const ts = document.getElementById('targetSelect');
          if (ts) {
            ts.value = d.detected_target;
            ts.dispatchEvent(new Event('change'));
          }
        }
        const ed = document.getElementById('emptyDropzone');
        if (ed) ed.style.display = 'none';
        const pacsActiveImage = document.getElementById('pacsActiveImage');
        if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(d.stored_path);
        const pic = document.getElementById('pacsImageContainer');
        if (pic) pic.style.display = 'flex';
        const ppc = document.getElementById('pacsPatientCode');
        if (ppc) ppc.textContent = "WEB_UPLOAD";
        const pd = document.getElementById('pacsDate');
        if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');
      }
    } catch (e) {
      alert('Không thể tải file lên: ' + e);
    }
  };
  inp.click();
}

function addAttachment(path, kind, name) {
  pendingAttachments.push({ path, kind, name });
  renderPreviews();
}

function renderPreviews() {
  const a = document.getElementById('imagePreviewArea');
  if (!a) return;
  a.innerHTML = '';
  if (pendingAttachments.length === 0) {
    a.classList.add('hidden');
    return;
  }
  a.classList.remove('hidden');
  pendingAttachments.forEach((att, i) => {
    const d = document.createElement('div');
    d.className = 'preview-thumb';
    const src = attachmentUrl(att.path);
    d.innerHTML = `<img src="${src}"><button class="remove-btn" onclick="removeAttachment(${i})">✕</button>`;
    a.appendChild(d);
  });
}

function removeAttachment(i) {
  pendingAttachments.splice(i, 1);
  renderPreviews();
  if (pendingAttachments.length === 0) {
    syncPACSView();
  }
}

async function triggerAIScreening() {
  if (isAnalyzing) return;
  const input = document.getElementById('messageInput');
  const targetSelect = document.getElementById('targetSelect');
  const modalitySelect = document.getElementById('modalitySelect');
  const target = targetSelect ? targetSelect.value : '';
  const modality = modalitySelect ? modalitySelect.value : '';
  const prompt = (input && input.value.trim()) || `Hãy sàng lọc y khoa cho ảnh này với nhóm bệnh ${target} bằng modality ${modality}.`;

  if (pendingAttachments.length === 0) {
    const pacsActiveImage = document.getElementById('pacsActiveImage');
    const src = pacsActiveImage ? (pacsActiveImage.getAttribute('src') || '') : '';
    if (src && !src.endsWith('/')) {
      const rel = src.startsWith('/output/') ? src.slice('/output/'.length) : decodeURIComponent(src.replace(/^\//, ''));
      pendingAttachments.push({
        path: rel,
        kind: 'image',
        name: rel.split('/').pop()
      });
    }
  }

  if (pendingAttachments.length > 0) {
    sendMessage();
  } else {
    alert('Vui lòng tải phim chụp lên trước khi yêu cầu phân tích AI.');
  }
}

async function sendMessage() {
  if (isAnalyzing) return;
  const input = document.getElementById('messageInput');
  const text = input ? input.value.trim() : '';
  if (!text && !pendingAttachments.length) return;

  const c = conversations[activeConvIndex];
  if (!c.id) {
    try {
      const r = await fetch('/api/conversations', { method: 'POST' });
      const d = await r.json();
      c.id = d.conversation_id;
    } catch (e) {}
  }

  if (text && c.title === t('new_chat')) {
    const fl = text.split('\n')[0].trim();
    if (fl && fl.length > 2) c.title = fl.substring(0, 28);
  }

  for (let i = 0; i < pendingAttachments.length; i++) {
    const att = pendingAttachments[i];
    const msgText = i === 0 && text ? text : '';
    const fd = new URLSearchParams();
    fd.append('sender', 'user');
    fd.append('text', msgText);
    fd.append('attachment_path', att.path);
    fd.append('attachment_kind', att.kind);
    if (c.id) {
      await fetch(`/api/conversations/${c.id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: fd
      }).catch(() => {});
    }
    c.messages.push({ sender: 'user', text: msgText, attachment_path: att.path, attachment_kind: att.kind });
  }

  if (text && !pendingAttachments.length) {
    const fd = new URLSearchParams();
    fd.append('sender', 'user');
    fd.append('text', text);
    if (c.id) {
      await fetch(`/api/conversations/${c.id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: fd
      }).catch(() => {});
    }
    c.messages.push({ sender: 'user', text: text });
  }

  if (input) {
    input.value = '';
    input.style.height = 'auto';
  }

  renderMessages();
  const atts = [...pendingAttachments];
  pendingAttachments = [];
  renderPreviews();

  const statusEl = document.getElementById('medicalStatus');
  const origStatusText = statusEl ? statusEl.textContent : '';
  if (statusEl) statusEl.textContent = t('medical_analyzing');

  isAnalyzing = true;
  showAILoadingIndicator();

  try {
    if (atts.length > 0) {
      const att = atts[0];
      const prompt = text || `Phân tích ảnh: ${att.name}`;
      const fd = new URLSearchParams();
      fd.append('image_path', att.path);
      fd.append('user_prompt', prompt);
      if (c.id) fd.append('conversation_id', c.id);

      let analysisOk = false;
      try {
        const r = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: fd
        });
        const d = await r.json();
        if (d.ok) {
          hideAILoadingIndicator();
          const metaJson = d.metadata ? JSON.stringify(d.metadata) : null;
          addBubble({
            sender: 'assistant',
            text: d.reply_text,
            attachment_path: d.attachment_path,
            attachment_kind: d.attachment_kind,
            metadata_json: metaJson
          }, true);

          if (c.id) {
            const fd2 = new URLSearchParams();
            fd2.append('sender', 'assistant');
            fd2.append('text', d.reply_text);
            fd2.append('attachment_path', d.attachment_path || '');
            fd2.append('attachment_kind', d.attachment_kind || '');
            fd2.append('metadata_json', metaJson || '');
            await fetch(`/api/conversations/${c.id}/messages`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
              body: fd2
            }).catch(() => {});
          }

          c.messages.push({
            sender: 'assistant',
            text: d.reply_text,
            attachment_path: d.attachment_path,
            attachment_kind: d.attachment_kind,
            metadata_json: metaJson
          });
          analysisOk = true;
        }
      } catch (e) {}

      if (!analysisOk) {
        hideAILoadingIndicator();
        const fname = att.name || att.path.split('/').pop().split('\\').pop();
        addBubble({ sender: 'assistant', text: 'Đã nhận ảnh: ' + fname, attachment_path: att.path, attachment_kind: att.kind }, true);
        if (c.id) {
          const fd2 = new URLSearchParams();
          fd2.append('sender', 'assistant');
          fd2.append('text', 'Đã nhận ảnh: ' + fname);
          fd2.append('attachment_path', att.path);
          fd2.append('attachment_kind', att.kind);
          await fetch(`/api/conversations/${c.id}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: fd2
          }).catch(() => {});
        }
        c.messages.push({ sender: 'assistant', text: 'Đã nhận ảnh: ' + fname, attachment_path: att.path, attachment_kind: att.kind });
      }
    } else {
      hideAILoadingIndicator();
      addBubble({ sender: 'assistant', text: t('system_reply_text') }, true);
      if (c.id) {
        const fd2 = new URLSearchParams();
        fd2.append('sender', 'assistant');
        fd2.append('text', t('system_reply_text'));
        await fetch(`/api/conversations/${c.id}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: fd2
        }).catch(() => {});
      }
      c.messages.push({ sender: 'assistant', text: t('system_reply_text') });
    }
  } finally {
    isAnalyzing = false;
    hideAILoadingIndicator();
    if (statusEl) statusEl.textContent = origStatusText;
    renderHistory();
    syncPACSView();
  }
}

function autoResize(el) {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function fillPrompt(txt) {
  const input = document.getElementById('messageInput');
  if (input) {
    input.value = txt;
    autoResize(input);
  }
}

function toggleVoice() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert('Trình duyệt không hỗ trợ nhận diện giọng nói.');
    return;
  }
  if (isRecording) {
    if (recognition) recognition.stop();
    return;
  }
  isRecording = true;
  const recPanel = document.getElementById('recordingPanel');
  if (recPanel) recPanel.classList.add('active');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = currentLang === 'vi' ? 'vi-VN' : 'en-US';
  recognition.interimResults = true;
  recognition.onresult = e => {
    const txt = Array.from(e.results).map(r => r[0].transcript).join('');
    const input = document.getElementById('messageInput');
    if (input) input.value = txt;
  };
  recognition.onerror = () => { stopVoice(); };
  recognition.onend = () => {
    stopVoice();
    const input = document.getElementById('messageInput');
    const txt = input ? input.value.trim() : '';
    if (txt) sendMessage();
  };
  recognition.start();
}

function stopVoice() {
  isRecording = false;
  const recPanel = document.getElementById('recordingPanel');
  if (recPanel) recPanel.classList.remove('active');
  if (recognition) {
    try { recognition.stop(); } catch (e) {}
    recognition = null;
  }
}

function openCamera() {
  const m = document.getElementById('plusMenu');
  if (m) m.style.display = 'none';
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = 'image/*';
  inp.capture = 'environment';
  inp.onchange = async () => {
    if (!inp.files[0]) return;
    const fd = new FormData();
    fd.append('file', inp.files[0]);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.ok) {
        addAttachment(d.stored_path, 'camera', '📷 ' + inp.files[0].name);
        const ed = document.getElementById('emptyDropzone');
        if (ed) ed.style.display = 'none';
        const pacsActiveImage = document.getElementById('pacsActiveImage');
        if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(d.stored_path);
        const pic = document.getElementById('pacsImageContainer');
        if (pic) pic.style.display = 'flex';
        const ppc = document.getElementById('pacsPatientCode');
        if (ppc) ppc.textContent = "CAMERA_CAPTURE";
        const pd = document.getElementById('pacsDate');
        if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');
      }
    } catch (e) {
      alert('Không thể lưu ảnh camera: ' + e);
    }
  };
  inp.click();
}

function previewImage(src) {
  const fullImg = document.getElementById('previewFullImg');
  const modal = document.getElementById('imagePreviewModal');
  if (fullImg) fullImg.src = src;
  if (modal) {
    modal.style.display = 'flex';
    modal.classList.add('active');
  }
}

function closeImagePreview() {
  const modal = document.getElementById('imagePreviewModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
}

// Drag & Drop Setup
const pacsScreen = document.getElementById('pacsScreen');
if (pacsScreen) {
  pacsScreen.addEventListener('dragenter', e => { e.preventDefault(); pacsScreen.classList.add('dragover'); });
  pacsScreen.addEventListener('dragover', e => { e.preventDefault(); pacsScreen.classList.add('dragover'); });
  pacsScreen.addEventListener('dragleave', () => { pacsScreen.classList.remove('dragover'); });
  pacsScreen.addEventListener('drop', async e => {
    e.preventDefault();
    pacsScreen.classList.remove('dragover');
    for (const f of e.dataTransfer.files) {
      const fd = new FormData();
      fd.append('file', f);
      try {
        const r = await fetch('/api/upload', { method: 'POST', body: fd });
        const d = await r.json();
        if (d.ok) {
          addAttachment(d.stored_path, 'image', f.name);
          if (d.detected_target) {
            const ts = document.getElementById('targetSelect');
            if (ts) {
              ts.value = d.detected_target;
              ts.dispatchEvent(new Event('change'));
            }
          }
          const ed = document.getElementById('emptyDropzone');
          if (ed) ed.style.display = 'none';
          const pacsActiveImage = document.getElementById('pacsActiveImage');
          if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(d.stored_path);
          const pic = document.getElementById('pacsImageContainer');
          if (pic) pic.style.display = 'flex';
          const ppc = document.getElementById('pacsPatientCode');
          if (ppc) ppc.textContent = "DRAG_DROP";
          const pd = document.getElementById('pacsDate');
          if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');
        }
      } catch (err) {}
    }
  });
}

document.addEventListener('dragover', e => { e.preventDefault(); });
document.addEventListener('drop', e => { e.preventDefault(); });

// Comparison Logic
let compareSelection = [];
async function openCompare(preId) {
  const m = document.getElementById('compareModal');
  if (m) m.classList.add('active');
  const pv = document.getElementById('comparePickView');
  const rv = document.getElementById('compareResultView');
  if (pv) pv.style.display = 'block';
  if (rv) rv.style.display = 'none';
  compareSelection = preId ? [preId] : [];
  try {
    const r = await fetch('/api/cases');
    const d = await r.json();
    renderCasePickList(d.cases || []);
  } catch (e) {
    renderCasePickList([]);
  }
}

function renderCasePickList(cases) {
  const l = document.getElementById('casePickList');
  const goBtn = document.getElementById('compareGoBtn');
  if (!l) return;
  if (!cases.length) {
    l.innerHTML = '<p style="font-size:13px;color:var(--text-muted);padding:12px">Chưa có ca bệnh nào. Hãy phân tích ảnh trước.</p>';
    if (goBtn) goBtn.disabled = true;
    return;
  }
  l.innerHTML = '';
  cases.forEach(c => {
    const d = document.createElement('div');
    d.className = 'case-pick-item' + (compareSelection.includes(c.case_id) ? ' selected' : '');
    const nDet = (c.detections || []).length;
    d.innerHTML = `
      <img src="${attachmentUrl(c.processed_image_path || c.image_path)}" loading="lazy">
      <div>
        <b>#${c.case_id} ${escHtml(c.patient_code)}</b>
        <div style="font-size:11px;color:var(--text-muted)">${escHtml(c.created_at || '')} • ${nDet} vùng</div>
      </div>
      <span class="risk-badge risk-${(c.risk_level || 'uncertain').toLowerCase()}" style="margin-left:auto">${escHtml(c.risk_level || '')}</span>
    `;
    d.onclick = () => {
      const i = compareSelection.indexOf(c.case_id);
      if (i >= 0) compareSelection.splice(i, 1);
      else {
        if (compareSelection.length >= 2) compareSelection.shift();
        compareSelection.push(c.case_id);
      }
      renderCasePickList(cases);
    };
    l.appendChild(d);
  });
  if (goBtn) goBtn.disabled = compareSelection.length !== 2;
}

function compareColHtml(c) {
  if (!c) return '<div class="compare-col"><p>Dữ liệu trống</p></div>';
  const risk = (c.risk_level || 'uncertain').toLowerCase();
  const row = (k, v) => `<div class="compare-row"><span>${k}</span><b>${v}</b></div>`;
  return `
    <div class="compare-col">
      <h3>#${c.case_id} ${escHtml(c.patient_code)} <span class="risk-badge risk-${risk}">${escHtml(risk)}</span></h3>
      <img src="${attachmentUrl(c.processed_image_path || c.image_path)}" onclick="previewImage(this.src)" loading="lazy">
      <div style="margin-top:8px">
        ${row('Tin cậy TB', ((c.average_confidence || 0) * 100).toFixed(1) + '%')}
        ${row('Vùng phát hiện', (c.detections || []).length)}
        ${row('Nghi ác tính', c.suspected_malignant ? 'Có' : 'Không')}
        ${row('Model', escHtml(c.model_name || '-'))}
      </div>
      <a class="result-btn" href="/api/cases/${c.case_id}/pdf" download style="margin-top:8px;display:block">⬇ Xuất PDF</a>
    </div>
  `;
}

async function renderCompare() {
  if (compareSelection.length !== 2) return;
  try {
    const results = await Promise.all(compareSelection.map(id => fetch('/api/cases/' + id).then(r => r.json())));
    const [a, b] = results.map(r => r.case);
    const view = document.getElementById('compareResultView');
    if (view) {
      view.innerHTML = `
        <div class="compare-grid">${compareColHtml(a)}${compareColHtml(b)}</div>
        <div style="text-align:center;margin-top:12px">
          <button class="result-btn" onclick="openCompare()" style="display:inline-block;min-width:160px">← Chọn lại</button>
        </div>
      `;
      const pv = document.getElementById('comparePickView');
      if (pv) pv.style.display = 'none';
      view.style.display = 'block';
    }
  } catch (e) {
    alert('Không thể so sánh ca bệnh: ' + e);
  }
}

// Global modal background click handler
document.addEventListener('click', e => {
  const settingsModal = document.getElementById('settingsModal');
  const compareModal = document.getElementById('compareModal');
  const previewModal = document.getElementById('imagePreviewModal');
  if (settingsModal && e.target === settingsModal) closeSettings();
  if (compareModal && e.target === compareModal) compareModal.classList.remove('active');
  if (previewModal && e.target === previewModal) closeImagePreview();
});

// App Initialization
(async function initApp() {
  const msgInput = document.getElementById('messageInput');
  if (msgInput) {
    msgInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  try {
    const r = await fetch('/api/settings');
    const d = await r.json();
    if (d.ok) {
      currentLang = d.language || 'vi';
      currentTheme = d.theme || 'system';
    }
  } catch (e) {}

  localStorage.setItem('lang', currentLang);
  localStorage.setItem('theme', currentTheme);

  const ts = document.getElementById('themeSetting');
  if (ts) ts.value = currentTheme;
  const ls = document.getElementById('langSetting');
  if (ls) ls.value = currentLang;

  initTargets();
  setTheme(currentTheme);
  applyLanguage(currentLang);
  await loadConversations();

  try {
    const r2 = await fetch('/api/status');
    const d2 = await r2.json();
    if (d2.ok) {
      const ms = document.getElementById('medicalStatus');
      if (ms) ms.textContent = 'Model: ' + (d2.model_ready ? 'Sẵn sàng' : 'Chưa sẵn sàng');
      const mb = document.getElementById('modeBadge');
      if (mb) mb.textContent = 'Y khoa • ' + (d2.analyzed_cancers || []).length + ' bệnh';
    }
  } catch (e) {}
})();
