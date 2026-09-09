const $ = (id) => document.getElementById(id);
const stateLabels = {
  outside: "Por llegar", waiting: "Fila general", validating: "Validando",
  ready_to_vote: "En carriles", voting: "Votando", finished: "Finalizados",
  rejected: "Rechazados", abandoned: "Abandonos", abstention: "Abstencion"
};

let finishedNoticeShown = false;
const partyColors = ["#2477b3", "#21845d", "#d8a82f", "#7351a3", "#bc4b45"];

function setText(id, value) { $(id).textContent = value; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }

function renderLanes(containerId, entries) {
  const container = $(containerId);
  container.innerHTML = entries.map(([label, value]) => `
    <div class="lane-row">
      <div class="lane-copy">
        <span>${label}</span><strong>${value} / 8</strong>
      </div>
      <div class="lane-track"><span class="lane-fill" style="width:${clamp(value / 8 * 100, 0, 100)}%"></span></div>
    </div>`).join("");
}

function renderPartyPie(entries) {
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  setText("partyVoteTotal", total);

  let accumulated = 0;
  const slices = entries.map(([, value], index) => {
    const start = accumulated;
    accumulated += total ? value / total * 100 : 0;
    return `${partyColors[index]} ${start}% ${accumulated}%`;
  });
  $("partyPie").style.background = total
    ? `conic-gradient(${slices.join(", ")})`
    : "#e1e6e2";

  $("partyLegend").innerHTML = entries.map(([party, value], index) => {
    const percentage = total ? value / total * 100 : 0;
    return `<div class="party-legend-row">
      <span class="party-swatch" style="background:${partyColors[index]}"></span>
      <b>${party}</b>
      <strong>${percentage.toFixed(1)}% <small>(${value})</small></strong>
    </div>`;
  }).join("");
  $("partyPie").setAttribute(
    "aria-label",
    total ? entries.map(([party, value]) => `${party}: ${value} votos`).join(", ") : "Todavia no hay votos validos"
  );
}

function renderResources(containerId, resources, labelPrefix) {
  const maxValue = Math.max(1, ...resources.map(item => item.processed));
  $(containerId).innerHTML = resources.map((item, index) => `
    <div class="resource-row">
      <div class="resource-copy"><span>${labelPrefix} ${index + 1}</span><span><b>${item.processed}</b> procesados · <span class="resource-state">${item.state}</span></span></div>
      <div class="resource-track"><span class="resource-fill" style="width:${item.processed / maxValue * 100}%"></span></div>
    </div>`).join("");
}

function renderFlowChart(history) {
  const canvas = $("flowChart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 38, right: 12, top: 10, bottom: 28 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const laneTotals = history.map(point => point.lane_queue_lengths.reduce((a, b) => a + b, 0));
  const maxY = Math.max(10, ...history.map(point => point.queue_length), ...laneTotals, ...history.map(point => point.finished_voters));

  ctx.strokeStyle = "#e1e5e1";
  ctx.fillStyle = "#6c7672";
  ctx.font = "10px ui-monospace, monospace";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    const label = Math.round(maxY * (1 - i / 4));
    ctx.fillText(String(label), 5, y + 3);
  }

  const series = [
    { color: "#2477b3", values: history.map(p => p.queue_length) },
    { color: "#7351a3", values: laneTotals },
    { color: "#21845d", values: history.map(p => p.finished_voters) }
  ];
  series.forEach(({ color, values }) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    values.forEach((value, index) => {
      const x = pad.left + (values.length <= 1 ? 0 : index / (values.length - 1)) * plotW;
      const y = pad.top + plotH - value / maxY * plotH;
      index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#6c7672";
  ctx.fillText("Inicio", pad.left, height - 6);
  ctx.textAlign = "right";
  ctx.fillText(`Paso ${history.at(-1)?.step || 0}`, width - pad.right, height - 6);
  ctx.textAlign = "left";
}

function renderDashboard(data) {
  const s = data.stats;
  const laneTotal = s.ballot_box_queue_lengths.reduce((a, b) => a + b, 0);
  const processed = s.completed_processes;
  const progress = s.total_turnout ? clamp(processed / s.total_turnout * 100, 0, 100) : 0;

  setText("currentEvent", data.current_event);
  setText("simulationClock", Number(data.simulation_clock).toFixed(2));
  setText("stepCount", data.step);
  setText("queueLength", s.queue_length);
  setText("queueMax", `Maximo: ${s.max_queue_length}`);
  setText("laneTotal", laneTotal);
  setText("finishedVoters", s.finished_voters);
  setText("activeVoters", `${s.active_voters} activos`);
  setText("averageWait", Number(s.average_waiting_time).toFixed(1));
  setText("p95Wait", `P95: ${Number(s.p95_waiting_time).toFixed(1)} min`);
  setText("abandonments", s.abandonment_count);
  setText("rejectedVoters", `${s.rejected_voters} rechazados`);
  setText("validVotes", s.valid_votes);
  setText("nullVotes", s.null_votes);
  setText("progressLabel", `${processed} de ${s.total_turnout} participantes procesados`);
  setText("progressPercent", `${progress.toFixed(0)}%`);
  $("progressBar").style.width = `${progress}%`;

  const partyEntries = Object.entries(s.votes_by_party);
  renderPartyPie(partyEntries);
  renderLanes("laneChart", s.ballot_box_queue_lengths.map((value, index) => [`Carril ${index + 1}`, value]));
  renderResources("workerTable", s.poll_worker_metrics, "Mesa");
  renderResources("boxTable", s.ballot_box_metrics, "Urna");

  const turnoutPercent = s.total_registered ? s.total_turnout / s.total_registered * 100 : 0;
  setText("turnoutPercent", `${turnoutPercent.toFixed(0)}%`);
  $("turnoutDonut").style.background = `conic-gradient(var(--green) ${turnoutPercent}%, #e1e6e2 ${turnoutPercent}%)`;
  const shownStates = ["waiting", "validating", "ready_to_vote", "voting", "finished", "rejected", "abandoned", "abstention"];
  $("stateBreakdown").innerHTML = shownStates.map(state => `<div class="state-item"><span>${stateLabels[state]}</span><strong>${s.state_counts[state] || 0}</strong></div>`).join("");

  const powerOut = s.power_status === "outage";
  setText("powerStatus", powerOut ? "Corte de energia" : "Energia normal");
  $("powerStatus").classList.toggle("outage", powerOut);
  setText("powerOutageStart", s.power_outage_started_at == null ? "--" : `t=${Number(s.power_outage_started_at).toFixed(2)}`);
  setText("powerRestoredAt", s.power_restored_at == null ? "--" : `t=${Number(s.power_restored_at).toFixed(2)}`);
  const outageTime = s.power_outage_duration ?? s.power_outage_elapsed;
  setText(
    "powerOutageDuration",
    outageTime == null ? "--" : `${Number(outageTime).toFixed(2)} min${powerOut ? " (en curso)" : ""}`
  );
  $("eventLog").innerHTML = data.recent_events.slice().reverse().map(event => `
    <div class="event-row"><span class="event-time">t=${Number(event.time).toFixed(2)}</span><span class="event-type">${event.event}</span><span>Agente ${event.voter_id ?? "sistema"}</span></div>`).join("") || '<div class="event-row"><span>Sin eventos</span></div>';

  renderFlowChart(data.history);
  setText("runState", data.is_finished ? "Simulacion terminada" : "Datos en vivo");
  setText("lastUpdate", `Actualizado ${new Date().toLocaleTimeString("es-MX")}`);
  $("connectionDot").className = `status-dot ${data.is_finished ? "" : "online"}`;
  if (data.is_finished && !finishedNoticeShown) {
    $("finishedBanner").classList.add("visible");
    finishedNoticeShown = true;
  }
}

async function refreshDashboard() {
  try {
    const response = await fetch("/dashboard_data?max_points=700", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderDashboard(await response.json());
  } catch (error) {
    $("connectionDot").className = "status-dot offline";
    setText("runState", "Sin conexion");
    setText("lastUpdate", "Inicia polling_station_api.py");
  }
}

$("resetButton").addEventListener("click", async () => {
  if (!window.confirm("¿Reiniciar toda la simulacion y borrar los resultados actuales?")) return;
  await fetch("/reset", { method: "POST" });
  finishedNoticeShown = false;
  $("finishedBanner").classList.remove("visible");
  refreshDashboard();
});
$("dismissBanner").addEventListener("click", () => $("finishedBanner").classList.remove("visible"));
window.addEventListener("resize", () => refreshDashboard());

refreshDashboard();
setInterval(refreshDashboard, 750);
