let chart;
let payload = null;

const statusEl = document.getElementById("status");
const subjectEl = document.getElementById("subject");
const loadBtn = document.getElementById("loadBtn");
const maxStepsEl = document.getElementById("maxSteps");
const slider = document.getElementById("stepSlider");

async function initSubjects() {
  const r = await fetch("/api/subjects");
  const d = await r.json();
  subjectEl.innerHTML = "";
  for (const id of d.subject_ids) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = `受试者 ${id}`;
    subjectEl.appendChild(o);
  }
  if (d.default_subject) {
    subjectEl.value = d.default_subject;
  }
}

function fmtArr(a, max = 8) {
  if (!a || !a.length) return "—";
  const head = a.slice(0, max);
  const tail = a.length > max ? ` …(共${a.length}点)` : "";
  return head.map((x) => Number(x).toFixed(0)).join(", ") + tail;
}

function updatePanels(idx) {
  if (!payload || !payload.steps.length) return;
  const s = payload.steps[idx];
  const L = payload.lookback;
  const H = payload.horizon;
  const g = payload.glucose_mgdl;
  const t = s.t;
  const past = g.slice(Math.max(0, t - L), t);
  const now = g[t];
  const futIdx = t + H;

  document.getElementById("ctxPast").textContent =
    `过去 ${L} 步 CGM（t−${L}…t−1，mg/dL）：\n` + fmtArr(past, 12);
  document.getElementById("ctxNow").textContent =
    `当前时刻 t=${t}：最新 CGM = ${Number(now).toFixed(1)} mg/dL（5 分钟一点）`;
  document.getElementById("ctxFuture").textContent =
    `目标时刻 t+${H}=${futIdx}：预测 = ${s.predicted_glucose.toFixed(1)} mg/dL · 实际 = ${s.actual_glucose.toFixed(1)} mg/dL`;

  const flags = document.getElementById("flags");
  flags.innerHTML = `
    <li class="${s.used_lstm ? "on" : "off"}">LSTM 预测工具：${s.used_lstm ? "已调用" : "未调用（沿用 cheap baseline）"}</li>
    <li class="${s.used_mc ? "on" : "off"}">MC dropout 不确定性：${s.used_mc ? "是" : "否"}</li>
    <li class="${s.used_guideline ? "on" : "off"}">指南/检索提示：${s.used_guideline ? "已展示" : "无"}</li>
    <li class="${s.anomaly_ood ? "on" : "off"}">异常检测 OOD：${s.anomaly_ood ? "是" : "否"}</li>
  `;
  document.getElementById("rationale").textContent = s.rationale || "—";
  const gq = document.getElementById("guideline");
  if (s.guideline_snippet) {
    gq.style.display = "block";
    gq.textContent = s.guideline_snippet;
  } else {
    gq.style.display = "none";
    gq.textContent = "";
  }

  document.getElementById("tInfo").textContent = `（${idx + 1} / ${payload.steps.length}）`;
}

function buildChart() {
  const g = payload.glucose_mgdl;
  const H = payload.horizon;
  const labels = g.map((_, i) => String(i));
  const predAt = new Array(g.length).fill(null);
  const actAt = new Array(g.length).fill(null);
  for (const s of payload.steps) {
    const j = s.t + H;
    if (j >= 0 && j < g.length) {
      predAt[j] = s.predicted_glucose;
      actAt[j] = s.actual_glucose;
    }
  }

  const ctx = document.getElementById("chart").getContext("2d");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "CGM（该受试者测试集）",
          data: g,
          borderColor: "rgba(37, 99, 235, 0.85)",
          backgroundColor: "rgba(37, 99, 235, 0.06)",
          fill: false,
          tension: 0.12,
          pointRadius: 0,
          borderWidth: 1.5,
        },
        {
          label: `模型/代理预测（对齐 t+${H}）`,
          data: predAt,
          borderColor: "rgba(234, 88, 12, 0.95)",
          showLine: false,
          pointRadius: (c) => (c.raw != null ? 4 : 0),
          pointBackgroundColor: "rgba(234, 88, 12, 0.95)",
        },
        {
          label: "同一时刻实际 CGM",
          data: actAt,
          borderColor: "rgba(22, 163, 74, 0.9)",
          showLine: false,
          pointRadius: (c) => (c.raw != null ? 3 : 0),
          pointBackgroundColor: "rgba(22, 163, 74, 0.85)",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "时间索引（5 min/步）" },
          ticks: { maxTicksLimit: 16 },
        },
        y: {
          title: { display: true, text: "血糖 mg/dL" },
        },
      },
      plugins: {
        legend: { position: "bottom" },
      },
    },
  });
}

slider.addEventListener("input", () => {
  const i = Number(slider.value);
  updatePanels(i);
});

loadBtn.addEventListener("click", async () => {
  statusEl.textContent = "加载中…";
  const sub = subjectEl.value;
  const maxSteps = Number(maxStepsEl.value) || 2500;
  try {
    const r = await fetch(
      `/api/trajectory?subject=${encodeURIComponent(sub)}&max_steps=${maxSteps}`
    );
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || r.statusText);
    }
    payload = await r.json();
    slider.max = String(Math.max(0, payload.steps.length - 1));
    slider.value = "0";
    buildChart();
    updatePanels(0);
    statusEl.textContent = `受试者 ${payload.subject} · 共 ${payload.steps.length} 个决策点`;
  } catch (err) {
    statusEl.textContent = "错误：" + err.message;
    console.error(err);
  }
});

initSubjects().catch(console.error);
