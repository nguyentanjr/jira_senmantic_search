"use strict";

const $ = (id) => document.getElementById(id);
const labels = { keyword: "Keyword (BM25)", semantic: "Semantic (e5)", hybrid: "Hybrid (BM25 + e5, RRF)" };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}
function safeJiraUrl(value) { try { const url = new URL(value); return url.protocol === "https:" ? url.href : "#"; } catch (_) { return "#"; } }
function toast(message) { const box = $("toast"); $("toast-text").textContent = message; box.classList.remove("hidden"); window.clearTimeout(toast.timer); toast.timer = window.setTimeout(() => box.classList.add("hidden"), 3500); }
function sharedKeys(results) { const count = new Map(); Object.values(results).flat().forEach((item) => count.set(item.key, (count.get(item.key) || 0) + 1)); return count; }
function card(item, count) {
  const shared = count.get(item.key) || 0;
  return `<article class="compare-card${shared >= 2 ? ` shared-${shared}` : ""}"><div class="compare-card-head"><span class="rank">${String(item.rank).padStart(2, "0")}</span><a href="${safeJiraUrl(item.url)}" target="_blank" rel="noopener">${escapeHtml(item.key)} ↗</a></div><h3>${escapeHtml(item.summary)}</h3><p>${escapeHtml(item.snippet || "")}</p></article>`;
}
function render(data) {
  const counts = sharedKeys(data.results);
  $("compare-results").innerHTML = Object.keys(labels).map((name) => {
    const timing = data.latency[name]; const items = data.results[name];
    return `<section class="compare-column"><h2>${labels[name]}</h2><p class="compare-latency">${timing.total_ms.toLocaleString("vi-VN")} ms · embedding ${timing.embed_ms.toLocaleString("vi-VN")} ms · tìm ${timing.search_ms.toLocaleString("vi-VN")} ms</p>${items.length ? items.map((item) => card(item, counts)).join("") : '<p class="compare-empty">Không có kết quả.</p>'}</section>`;
  }).join("");
  $("compare-results").classList.remove("hidden");
}
$("compare-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const query = $("query").value.trim(); const button = $("submit-button");
  button.disabled = true; button.querySelector("span").textContent = "Đang tìm…";
  try { const response = await fetch("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, top_k: 10 }) }); const data = await response.json(); if (!response.ok) throw new Error(data.error || "Không thể tìm kiếm."); render(data); }
  catch (error) { toast(error.message); }
  finally { button.disabled = false; button.querySelector("span").textContent = "Tìm"; }
});
$("query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("compare-form").requestSubmit();
  }
});
