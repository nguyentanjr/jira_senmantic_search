"use strict";

const REASONS = [
  ["wrong_module", "Sai module / luồng"],
  ["wrong_defect_type", "Sai loại lỗi"],
  ["opposite_direction", "Ngược chiều"],
  ["other", "Khác"]
];
const REASON_LABEL = Object.fromEntries(REASONS);

// Trang thai danh gia cua luot tim hien tai. Server ghi moi thao tac thanh su kien;
// UI chi phan anh nhan cuoi cung (doi y, bo chon deu duoc ghi).
const state = { flow: "text", meta: null, lastRequest: null, lastResponse: null,
  generation: 0,
  labels: new Map(),   // key -> {verdict, reason, note, rank} da luu
  pending: new Set(),  // key vua bam "Khong lien quan", dang cho chon nguyen nhan
  busy: new Set(),     // key dang gui request
  errors: new Map(),   // key -> thong bao loi gan nhat
  noAnswer: false, noAnswerBusy: false, suggestionTimer: null,
  feedbackEnabled: readFeedbackPreference() };

function readFeedbackPreference() {
  try { return window.localStorage.getItem("pilot.feedbackEnabled") !== "0"; }
  catch (_) { return true; }
}
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}

function safeJiraUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.href : "#";
  } catch (_) { return "#"; }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ error: "Phản hồi server không hợp lệ." }));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function showToast(message, error = false, action = null) {
  const toast = $("toast");
  const button = $("toast-action");
  $("toast-text").textContent = message;
  toast.className = `toast${error ? " error" : ""}`;
  button.classList.toggle("hidden", !action);
  button.onclick = null;
  if (action) {
    button.textContent = action.label;
    button.onclick = () => { toast.classList.add("hidden"); action.run(); };
  }
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.add("hidden"), action ? 7000 : 3500);
}

function applyFeedbackPreference() {
  document.body.classList.toggle("feedback-off", !state.feedbackEnabled);
  $("feedback-toggle").checked = state.feedbackEnabled;
}

function setFeedbackEnabled(enabled) {
  state.feedbackEnabled = enabled;
  try { window.localStorage.setItem("pilot.feedbackEnabled", enabled ? "1" : "0"); } catch (_) { /* bo qua */ }
  if (!enabled) state.pending.clear();  // chua luu gi nen bo di khong can hoi
  applyFeedbackPreference();
  paintAll();
  showToast(enabled ? "Đã bật đánh giá kết quả." : "Đã tắt đánh giá. Các đánh giá đã lưu vẫn được giữ.");
}

function fillSelect(id, values) {
  const select = $(id);
  values.forEach(({ value, count }) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = `${value} (${count.toLocaleString("vi-VN")})`;
    select.append(option);
  });
}

async function loadMeta() {
  try {
    state.meta = await api("/api/meta");
    const f = state.meta.filters;
    fillSelect("filter-project", f.projects);
    fillSelect("filter-component", f.components);
    fillSelect("filter-status", f.statuses);
    fillSelect("filter-priority", f.priorities);
    $("filter-from").min = f.created_min; $("filter-from").max = f.created_max;
    $("filter-to").min = f.created_min; $("filter-to").max = f.created_max;
  } catch (error) {
    showToast(error.message, true);
  }
}

function setFlow(flow) {
  state.flow = flow;
  document.querySelectorAll(".flow-tab").forEach((tab) => {
    const active = tab.dataset.flow === flow;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $("text-input-wrap").classList.toggle("hidden", flow !== "text");
  $("issue-input-wrap").classList.toggle("hidden", flow !== "issue");
  (flow === "text" ? $("query") : $("issue-key")).focus();
}

function filters() {
  return {
    project: $("filter-project").value,
    component: $("filter-component").value,
    status: $("filter-status").value,
    priority: $("filter-priority").value,
    created_from: $("filter-from").value,
    created_to: $("filter-to").value
  };
}

function badges(result) {
  const values = [result.status, result.priority, result.created, ...result.components.slice(0, 2)];
  return values.filter(Boolean).map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("");
}

function renderResult(result, request) {
  const resolution = result.resolution
    ? `<div class="resolution-box"><strong>Ghi nhận xử lý</strong>${escapeHtml(result.resolution)}</div>`
    : "";
  const resolvedBadge = result.resolution_status
    ? `<span class="badge resolved">${escapeHtml(result.resolution_status)}</span>` : "";
  return `<article class="result-card" data-key="${escapeHtml(result.key)}">
    <div class="rank">${String(result.rank).padStart(2, "0")}</div>
    <div class="result-main">
      <div class="result-title"><a href="${safeJiraUrl(result.url)}" target="_blank" rel="noopener">${escapeHtml(result.key)} ↗</a><h3>${escapeHtml(result.summary)}</h3></div>
      <div class="badges">${resolvedBadge}${badges(result)}</div>
      <p class="snippet">${escapeHtml(result.snippet)}</p>
      ${resolution}
      <div class="feedback" data-key="${escapeHtml(result.key)}" data-rank="${result.rank}">
        <div class="feedback-row">
          <span class="feedback-question">Kết quả này có liên quan không?</span>
          <div class="verdict-group" role="group" aria-label="Đánh giá ${escapeHtml(result.key)}">
            <button type="button" class="verdict related" data-action="related" aria-pressed="false">Có, liên quan</button>
            <button type="button" class="verdict not-related" data-action="not_related" aria-pressed="false">Không liên quan</button>
          </div>
          <div class="feedback-state" aria-live="polite">
            <span class="feedback-status"></span>
            <button type="button" class="clear-feedback hidden" data-action="clear">Xóa đánh giá</button>
          </div>
        </div>
        <div class="feedback-panel hidden">
          <div class="reason-block hidden">
            <div class="reason-heading"><span class="panel-label">Vì sao kết quả này không liên quan?</span><button type="button" class="cancel-feedback" data-action="cancel">Hủy</button></div>
            <div class="reason-group" role="group" aria-label="Nguyên nhân kết quả không liên quan">
              ${REASONS.map(([value, label]) => `<button type="button" class="reason-chip" data-reason="${value}" aria-pressed="false">${label}</button>`).join("")}
            </div>
          </div>
          <label class="note-field">Ghi chú <span>(tùy chọn)</span>
            <input class="feedback-note" maxlength="500" placeholder="Nhấn Enter hoặc rời ô để lưu">
          </label>
        </div>
      </div>
    </div>
  </article>`;
}

function feedbackBox(key) {
  return document.querySelector(`.feedback[data-key="${CSS.escape(key)}"]`);
}

function paintFeedback(key) {
  const box = feedbackBox(key);
  if (!box) return;
  const label = state.labels.get(key);
  const pending = state.pending.has(key);
  const busy = state.busy.has(key);
  const related = box.querySelector(".verdict.related");
  const negative = box.querySelector(".verdict.not-related");
  const relatedOn = label?.verdict === "related" && !pending;
  const negativeSaved = label?.verdict === "not_related" && !pending;
  related.classList.toggle("on", relatedOn);
  related.setAttribute("aria-pressed", String(relatedOn));
  negative.classList.toggle("on", negativeSaved);
  negative.classList.toggle("pending", pending);
  negative.setAttribute("aria-pressed", String(negativeSaved || pending));
  box.querySelectorAll("button").forEach((button) => { button.disabled = busy; });

  box.querySelector(".feedback-panel").classList.toggle("hidden", !label && !pending);
  box.querySelector(".reason-block").classList.toggle("hidden", !pending && !negativeSaved);
  box.querySelector('[data-action="cancel"]').classList.toggle("hidden", !pending);
  box.querySelectorAll(".reason-chip").forEach((chip) => {
    chip.setAttribute("aria-pressed", String(negativeSaved && label.reason === chip.dataset.reason));
  });
  box.querySelector(".feedback-note").placeholder = relatedOn
    ? "Không bắt buộc, vd: khác module nhưng cùng cách sửa · Enter để lưu"
    : "Không bắt buộc · Enter để lưu";

  const clear = box.querySelector('[data-action="clear"]');
  clear.classList.toggle("hidden", !label || pending);

  const status = box.querySelector(".feedback-status");
  let text = "";
  let tone = "";
  if (busy) { text = "Đang lưu…"; }
  else if (state.errors.has(key)) { text = `Chưa lưu: ${state.errors.get(key)}`; tone = "error"; }
  else if (pending) {
    text = label ? "Chọn lý do để thay đánh giá hiện tại" : "Chọn một lý do để hoàn tất đánh giá";
    tone = "warn";
  } else if (relatedOn) { text = "Đã đánh giá: liên quan"; tone = "ok"; }
  else if (negativeSaved) { text = `Đã đánh giá: ${REASON_LABEL[label.reason] || label.reason}`; tone = "ok"; }
  status.textContent = text;
  status.className = `feedback-status${tone ? " " + tone : ""}`;

  const card = box.closest(".result-card");
  card.classList.toggle("judged-related", relatedOn);
  card.classList.toggle("judged-negative", negativeSaved);
}

function paintNoAnswer() {
  const button = $("no-answer-button");
  const relatedCount = [...state.labels.values()].filter((label) => label.verdict === "related").length;
  button.classList.toggle("selected", state.noAnswer);
  button.setAttribute("aria-pressed", String(state.noAnswer));
  button.disabled = state.noAnswerBusy;
  button.textContent = state.noAnswer
    ? "✓ Đã ghi nhận: không có đáp án · bấm để bỏ"
    : relatedCount
      ? `Không có đáp án phù hợp · sẽ bỏ ${relatedCount} đánh giá liên quan`
      : "Không có đáp án phù hợp trong top 10";
}

function updateReviewProgress() {
  const total = state.lastResponse?.results?.length || 0;
  const judged = new Set([...state.labels.values()].map((label) => label.rank));
  const pendingNote = state.pending.size ? ` · ${state.pending.size} chưa chọn nguyên nhân` : "";
  $("feedback-count").textContent = total
    ? `Đã đánh giá ${judged.size}/${total} kết quả${pendingNote}`
    : "";
  const full = state.lastResponse?.review_mode === "full_top5";
  $("review-mode").classList.toggle("hidden", !full);
  document.querySelectorAll(".result-card").forEach((card) => {
    const rank = Number(card.querySelector(".feedback")?.dataset.rank);
    card.classList.toggle("needs-review", full && !state.noAnswer && rank <= 5 && !judged.has(rank));
  });
  if (!full) return;
  const target = Math.min(5, total);
  const missing = [1, 2, 3, 4, 5].filter((rank) => rank <= total && !judged.has(rank));
  const done = state.noAnswer || missing.length === 0;
  $("review-progress").textContent = state.noAnswer
    ? "Đã ghi nhận: không có đáp án trong top 10."
    : done ? `Đã chấm đủ ${target} kết quả đầu ✓`
      : `Đã chấm ${target - missing.length}/${target} · còn thiếu #${missing.join(", #")}`;
  $("review-mode").classList.toggle("complete", done);
}

function paintAll() {
  document.querySelectorAll(".feedback").forEach((box) => paintFeedback(box.dataset.key));
  paintNoAnswer();
  updateReviewProgress();
}

function renderResults(data, request) {
  state.lastResponse = data;
  state.generation += 1;
  state.labels.clear(); state.pending.clear(); state.busy.clear(); state.errors.clear();
  state.noAnswer = false; state.noAnswerBusy = false;
  $("welcome-state").classList.add("hidden");
  $("results-section").classList.remove("hidden");
  $("results-title").textContent = data.results.length ? "Issue tương tự" : "Chưa có kết quả";
  $("results-list").innerHTML = data.results.length
    ? data.results.map((result) => renderResult(result, request)).join("")
    : `<div class="no-results"><strong>Không có issue nào thỏa bộ lọc.</strong><p>Hãy nới một hoặc nhiều điều kiện rồi tìm lại.</p></div>`;
  $("query-note").value = "";
  $("query-feedback").classList.toggle("hidden", !data.results.length);
  paintAll();
  $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runSearch(event) {
  event.preventDefault();
  const request = { flow: state.flow, filters: filters(), top_k: 10,
    feedback_enabled: state.feedbackEnabled };
  if (state.flow === "text") request.query = $("query").value.trim();
  else request.key = $("issue-key").value.trim().toUpperCase();
  const button = $("submit-button");
  const old = button.querySelector("span").textContent;
  button.disabled = true; button.querySelector("span").textContent = "Đang xếp hạng…";
  try {
    const data = await api("/api/search", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request)
    });
    state.lastRequest = request;
    renderResults(data, request);
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.querySelector("span").textContent = old; }
}

function postFeedback(payload) {
  return api("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ search_id: state.lastResponse.search_id, ...payload })
  });
}

// Moi thao tac tren mot ket qua di qua day: khoa nut trong luc gui va bo qua
// phan hoi den muon neu nguoi dung da sang luot tim khac.
async function withBusy(key, work) {
  const generation = state.generation;
  const current = () => generation === state.generation;
  state.busy.add(key); state.errors.delete(key); paintFeedback(key);
  try {
    await work(current);
  } catch (error) {
    if (current()) state.errors.set(key, error.message);
    showToast(error.message, true);
  } finally {
    if (current()) { state.busy.delete(key); paintAll(); }
  }
}

function saveLabel(key, verdict, reason = "") {
  const box = feedbackBox(key);
  const note = box.querySelector(".feedback-note").value.trim();
  const rank = Number(box.dataset.rank);
  return withBusy(key, async (current) => {
    await postFeedback({ result_key: key, verdict, reason, note });
    if (!current()) return;
    state.labels.set(key, { verdict, reason, note, rank });
    state.pending.delete(key);
    // Server tu huy "khong co dap an" khi co ket qua lien quan.
    if (verdict === "related") state.noAnswer = false;
  });
}

function clearLabel(key) {
  const box = feedbackBox(key);
  if (!state.labels.has(key)) {
    state.pending.delete(key);
    paintAll();
    return Promise.resolve();
  }
  return withBusy(key, async (current) => {
    await postFeedback({ result_key: key, verdict: "cleared" });
    if (!current()) return;
    state.labels.delete(key);
    state.pending.delete(key);
    box.querySelector(".feedback-note").value = "";
  });
}

function onFeedbackClick(event) {
  const box = event.target.closest(".feedback");
  if (!box || !state.lastResponse) return;
  const key = box.dataset.key;
  if (state.busy.has(key)) return;
  const label = state.labels.get(key);
  const pending = state.pending.has(key);
  const chip = event.target.closest(".reason-chip");
  if (chip) {
    const reason = chip.dataset.reason;
    if (!pending && label?.verdict === "not_related" && label.reason === reason) return;
    saveLabel(key, "not_related", reason);
    return;
  }
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "related") {
    if (label?.verdict === "related" && !pending) return;
    else { state.pending.delete(key); saveLabel(key, "related"); }
  } else if (action === "not_related") {
    if (pending || label?.verdict === "not_related") return;
    else {
      state.pending.add(key);
      paintAll();
      box.querySelector(".reason-chip")?.focus();
    }
  } else if (action === "clear") {
    clearLabel(key);
  } else if (action === "cancel") {
    state.pending.delete(key);
    paintAll();
  }
}

function onFeedbackNoteChange(event) {
  if (!event.target.matches(".feedback-note")) return;
  const key = event.target.closest(".feedback").dataset.key;
  const label = state.labels.get(key);
  // Chua co nhan: ghi chu duoc gui kem lan danh gia tiep theo.
  if (!label || state.pending.has(key) || state.busy.has(key)) return;
  if (event.target.value.trim() === label.note) return;
  saveLabel(key, label.verdict, label.reason);
}

async function setNoAnswer(on, { quiet = false } = {}) {
  const generation = state.generation;
  state.noAnswerBusy = true;
  paintNoAnswer();
  try {
    await postFeedback({ result_key: "", verdict: on ? "no_answer" : "cleared",
      note: on ? $("query-note").value.trim() : "" });
    if (generation !== state.generation) return [];
    state.noAnswer = on;
    const dropped = [];
    if (on) {
      // Server bo cac nhan "lien quan" cua luot nay; UI lam giong the.
      for (const [key, label] of [...state.labels]) {
        if (label.verdict === "related") { dropped.push([key, label]); state.labels.delete(key); }
      }
    }
    if (!quiet) {
      if (on && dropped.length) {
        showToast(`Đã ghi nhận “không có đáp án” và bỏ ${dropped.length} đánh giá “liên quan”.`, false,
          { label: "Hoàn tác", run: () => undoNoAnswer(generation, dropped) });
      } else {
        showToast(on ? "Đã ghi nhận top 10 không có đáp án phù hợp." : "Đã bỏ đánh dấu “không có đáp án”.");
      }
    }
    return dropped;
  } catch (error) { showToast(error.message, true); return []; }
  finally {
    if (generation === state.generation) { state.noAnswerBusy = false; paintAll(); }
  }
}

async function undoNoAnswer(generation, dropped) {
  if (generation !== state.generation) return;
  await setNoAnswer(false, { quiet: true });
  for (const [key, label] of dropped) {
    if (generation !== state.generation) return;
    const box = feedbackBox(key);
    if (box) box.querySelector(".feedback-note").value = label.note || "";
    await saveLabel(key, "related");
  }
  showToast("Đã hoàn tác.");
}

function toggleNoAnswer() {
  if (!state.lastResponse || state.noAnswerBusy) return;
  setNoAnswer(!state.noAnswer);
}

async function suggestIssues() {
  const q = $("issue-key").value.trim();
  const box = $("suggestions");
  if (q.length < 2) { box.classList.add("hidden"); return; }
  try {
    const data = await api(`/api/issues?q=${encodeURIComponent(q)}&limit=8`);
    box.innerHTML = data.items.map((item) => `<button class="suggestion" type="button" role="option" data-key="${escapeHtml(item.key)}"><strong>${escapeHtml(item.key)}</strong><span>${escapeHtml(item.summary)}</span></button>`).join("");
    box.classList.toggle("hidden", !data.items.length);
  } catch (_) { box.classList.add("hidden"); }
}

document.querySelectorAll(".flow-tab").forEach((tab) => tab.addEventListener("click", () => setFlow(tab.dataset.flow)));
$("query").addEventListener("input", () => $("char-count").textContent = `${$("query").value.length.toLocaleString("vi-VN")} / 5.000`);
$("search-form").addEventListener("submit", runSearch);
$("clear-filters").addEventListener("click", () => document.querySelectorAll(".filter-panel select, .filter-panel input").forEach((field) => { field.value = ""; }));
$("issue-key").addEventListener("input", () => { window.clearTimeout(state.suggestionTimer); state.suggestionTimer = window.setTimeout(suggestIssues, 180); });
$("suggestions").addEventListener("click", (event) => { const item = event.target.closest(".suggestion"); if (item) { $("issue-key").value = item.dataset.key; $("suggestions").classList.add("hidden"); } });
$("results-list").addEventListener("click", onFeedbackClick);
$("results-list").addEventListener("change", onFeedbackNoteChange);
$("results-list").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.target.matches(".feedback-note")) event.target.blur();
  if (event.key === "Escape") {
    const box = event.target.closest(".feedback");
    if (box && state.pending.has(box.dataset.key)) { state.pending.delete(box.dataset.key); paintAll(); }
  }
});
$("no-answer-button").addEventListener("click", toggleNoAnswer);
$("query-note").addEventListener("change", () => { if (state.noAnswer && !state.noAnswerBusy) setNoAnswer(true); });
$("query-note").addEventListener("keydown", (event) => { if (event.key === "Enter") event.target.blur(); });
$("feedback-toggle").addEventListener("change", (event) => setFeedbackEnabled(event.target.checked));
document.addEventListener("click", (event) => { if (!event.target.closest(".issue-combobox")) $("suggestions").classList.add("hidden"); });

applyFeedbackPreference();
loadMeta();
