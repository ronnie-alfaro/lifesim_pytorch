const ACTIONS = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "EAT", "DRINK", "REST", "WAIT"];

const ui = Object.fromEntries([
  "world", "connection-dot", "run-label", "tick-label", "status-label", "tick-progress",
  "play-button", "pause-button", "step-button", "speed-input", "speed-label",
  "humans-alive", "animals-alive", "resources-count", "average-loss", "training-message",
  "completion-summary", "termination-label", "survivor-summary", "run-comparison", "learning-numbers", "learned-explanation",
  "next-run-button", "comparison-chart-wrap", "comparison-chart",
  "apply-config-button", "human-count-input", "human-count-label",
  "animal-count-input", "animal-count-label", "human-brain-input",
  "human-brain-label", "animal-brain-input", "animal-brain-label",
  "architecture-preview", "builder-help", "use-brb-input", "brb-description",
  "all-humans-button", "all-animals-button",
  "agent-name", "agent-kind", "agent-empty", "agent-details", "vitals", "agent-action",
  "agent-reward", "agent-loss", "agent-mode", "agent-replay", "agent-updates", "epsilon-explanation",
  "brain-network", "brain-network-caption", "brain-view-label", "spatial-memory",
  "q-values", "observations", "reward-components", "event-stream", "toast"
].map(id => [id, document.getElementById(id)]));

let state = null;
let selectedAgentId = null;
let selectedScope = "human";
let pollTimer = null;
let configurationDirty = false;
const previousWeightMeans = new Map();

async function api(path, options = {}) {
  const response = await fetch(path, {cache: "no-store", ...options});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function poll() {
  try {
    state = await api("/api/state");
    ui["connection-dot"].classList.add("online");
    render();
    schedulePoll(state.status === "running" ? 160 : 450);
  } catch (error) {
    ui["connection-dot"].classList.remove("online");
    showError(`Conexión perdida: ${error.message}`);
    schedulePoll(1000);
  }
}

function schedulePoll(delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, delay);
}

async function control(action, value = undefined) {
  try {
    state = await api("/api/control", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action, value})
    });
    render();
    schedulePoll(80);
  } catch (error) {
    showError(error.message);
  }
}

function render() {
  if (!state) return;
  ui["run-label"].textContent = `Experimento ${pad(state.experiment_id)} · Run ${pad(state.run_number)}`;
  ui["tick-label"].textContent = `Tick ${state.tick.toLocaleString()} / ${state.num_ticks.toLocaleString()}`;
  ui["tick-progress"].style.width = `${(state.tick / state.num_ticks) * 100}%`;
  ui["status-label"].textContent = statusText(state.status);
  ui["status-label"].className = `status-badge ${state.status}`;
  ui["play-button"].disabled = state.status !== "paused";
  ui["pause-button"].disabled = state.status !== "running";
  ui["step-button"].disabled = state.status !== "paused";
  ui["next-run-button"].disabled = !state.can_start_next_run;
  ui["next-run-button"].textContent = state.status === "preparing" ? "Preparando mundo…" : "↻ Siguiente ciclo";
  ui["speed-input"].value = state.ticks_per_second;
  ui["speed-label"].textContent = `${format(state.ticks_per_second, 0)} ticks/s`;
  renderExperimentBuilder();

  const summary = state.summary;
  ui["humans-alive"].textContent = `${summary.living_humans} / ${state.experiment_config.num_humans}`;
  ui["animals-alive"].textContent = `${summary.living_animals} / ${state.experiment_config.num_animals}`;
  ui["resources-count"].textContent = `${summary.food_remaining} / ${summary.water_remaining}`;
  ui["average-loss"].textContent = summary.average_loss == null ? "—" : format(summary.average_loss, 4);
  renderWorld();
  renderInspector();
  renderEvents();
  renderTrainingState();
  renderCompletionSummary();
  if (state.error) showError(state.error);
}

function renderExperimentBuilder() {
  const config = state.experiment_config;
  const brb = state.brb;
  if (!configurationDirty) {
    ui["human-count-input"].value = config.num_humans;
    ui["animal-count-input"].value = config.num_animals;
    ui["human-brain-input"].value = config.human_hidden_sizes.at(-1);
    ui["animal-brain-input"].value = config.animal_hidden_sizes.at(-1);
  }
  ui["use-brb-input"].disabled = !brb || !state.can_configure_experiment;
  if (!brb) ui["use-brb-input"].checked = false;
  applyBrbMinimums();
  updateBuilderLabels();
  ui["apply-config-button"].disabled = !state.can_configure_experiment;
  const completion = brb?.final_humans == null
    ? ""
    : ` · ${brb.final_humans}/${brb.initial_humans} humanos completaron el ciclo`;
  ui["brb-description"].textContent = brb
    ? `Campeón: experimento ${pad(brb.experiment_id)}, run ${pad(brb.run_number)}${completion} · supervivencia media ${format(brb.mean_human_survival, 1)}, mediana ${format(brb.median_human_survival, 1)}.${brb.learning_contract_current ? "" : ` Pesos base v${brb.source_reward_version}; reward v${brb.current_reward_version} reinicia Adam/replay y solo lo reemplazará un resultado compatible mejor.`}`
    : "Todavía no existe un run completo compatible para usar como campeón.";
  ui["builder-help"].textContent = state.can_configure_experiment
    ? (ui["use-brb-input"].checked
      ? "Copia los pesos del campeón. Mundo, cuerpos, Adam y replay Horde comienzan limpios para comparar el nuevo experimento."
      : "Crea un experimento independiente con brains nuevos. Más agentes y neuronas requieren más CPU.")
    : "Bloqueado durante el run. Termina el ciclo para crear otro experimento; Siguiente ciclo conserva los brains actuales.";
}

function applyBrbMinimums() {
  const brb = state?.brb;
  const enabled = Boolean(brb && ui["use-brb-input"].checked);
  const humanMinimum = enabled ? Number(brb.minimum_human_width) : 8;
  const animalMinimum = enabled ? Number(brb.minimum_animal_width) : 8;
  ui["human-brain-input"].min = humanMinimum;
  ui["animal-brain-input"].min = animalMinimum;
  if (Number(ui["human-brain-input"].value) < humanMinimum) ui["human-brain-input"].value = humanMinimum;
  if (Number(ui["animal-brain-input"].value) < animalMinimum) ui["animal-brain-input"].value = animalMinimum;
}

function updateBuilderLabels() {
  const humans = Number(ui["human-count-input"].value);
  const animals = Number(ui["animal-count-input"].value);
  const humanWidth = Number(ui["human-brain-input"].value);
  const animalWidth = Number(ui["animal-brain-input"].value);
  ui["human-count-label"].textContent = humans;
  ui["animal-count-label"].textContent = animals;
  ui["human-brain-label"].textContent = `${humanWidth} neuronas/capa`;
  ui["animal-brain-label"].textContent = `${animalWidth} neuronas`;
  ui["architecture-preview"].textContent = `Humano: supervivencia 15→${Math.max(8, humanWidth / 2)}, espacio 18→${humanWidth}, fusión ${humanWidth}→8 · Animal: supervivencia 15, espacio 16→${animalWidth} · ${humans + animals} brains v2`;
}

function renderWorld() {
  const {width, height, food, water, obstacles} = state.grid;
  const canvas = ui.world;
  // Render at a larger native pixel resolution, then let CSS fit the same panel.
  // This keeps every tiny sprite crisp on high-density displays.
  const cellSize = 16;
  canvas.width = width * cellSize;
  canvas.height = height * cellSize;
  canvas.style.aspectRatio = `${width} / ${height}`;
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = false;

  // The visual world is an explicit matrix: 0 terrain, 1 obstacle, 2 food, 3 water.
  const matrix = Array.from({length: height}, () => new Uint8Array(width));
  for (const [x, y] of obstacles) matrix[y][x] = 1;
  for (const [x, y] of food) matrix[y][x] = 2;
  for (const [x, y] of water) matrix[y][x] = 3;

  context.fillStyle = "#dfe1cf";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) drawMatrixCell(context, matrix[y][x], x, y, cellSize, matrix);
  }
  for (const agent of state.agents) drawPixelAgent(context, agent, cellSize, agent.id === selectedAgentId);
}

function drawMatrixCell(context, value, x, y, size, matrix) {
  const px = x * size;
  const py = y * size;
  const terrain = (x + y) % 2 === 0 ? "#dfe2cf" : "#d9ddc8";
  context.fillStyle = terrain;
  context.fillRect(px, py, size, size);
  context.fillStyle = "rgba(75, 99, 69, .16)";
  if ((x * 7 + y * 11) % 13 === 0) context.fillRect(px + 3, py + 11, 2, 1);
  if (value === 1) {
    context.fillStyle = "rgba(38,49,43,.18)";
    context.fillRect(px + 3, py + 12, 11, 3);
    context.fillStyle = "#5e6861";
    context.fillRect(px + 2, py + 6, 12, 7);
    context.fillRect(px + 4, py + 3, 8, 10);
    context.fillStyle = "#879087";
    context.fillRect(px + 5, py + 4, 6, 2);
    context.fillRect(px + 3, py + 7, 3, 2);
  } else if (value === 2) {
    context.fillStyle = "#6e5937";
    context.fillRect(px + 4, py + 12, 9, 2);
    context.fillStyle = "#357143";
    context.fillRect(px + 7, py + 4, 2, 9);
    context.fillStyle = "#75a94e";
    context.fillRect(px + 3, py + 5, 5, 4);
    context.fillRect(px + 9, py + 3, 4, 5);
    context.fillStyle = "#d99a3f";
    context.fillRect(px + 10, py + 8, 3, 3);
  } else if (value === 3) {
    context.fillStyle = "#2879a4";
    context.fillRect(px, py, size, size);
    context.fillStyle = (x + y) % 3 === 0 ? "#78c4dd" : "#53a8ca";
    context.fillRect(px + 1, py + 4, 7, 2);
    context.fillRect(px + 9, py + 11, 6, 2);
    // A pale shoreline makes large permanent ponds readable as clusters.
    context.fillStyle = "#a8d9dd";
    if (y === 0 || matrix[y - 1][x] !== 3) context.fillRect(px, py, size, 1);
    if (x === 0 || matrix[y][x - 1] !== 3) context.fillRect(px, py, 1, size);
  }
}

function drawPixelAgent(context, agent, size, selected) {
  const px = agent.x * size;
  const py = agent.y * size;
  if (selected) {
    context.fillStyle = "rgba(255,246,179,.55)";
    context.fillRect(px, py, size, size);
    context.strokeStyle = "#fff8bd";
    context.lineWidth = 2;
    context.strokeRect(px + 1, py + 1, size - 2, size - 2);
  }
  if (!agent.alive) {
    context.strokeStyle = "rgba(60, 60, 57, .55)";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(px + 4, py + 4);
    context.lineTo(px + size - 4, py + size - 4);
    context.moveTo(px + size - 4, py + 4);
    context.lineTo(px + 4, py + size - 4);
    context.stroke();
    return;
  }
  if (agent.type === "human") {
    // Vertical pixel person: warm skin, bright torso, arms and separate legs.
    context.fillStyle = "rgba(52,38,27,.22)";
    context.fillRect(px + 5, py + 14, 7, 1);
    context.fillStyle = "#f2b06b";
    context.fillRect(px + 6, py + 1, 5, 5);
    context.fillStyle = "#7f3f2b";
    context.fillRect(px + 6, py + 1, 5, 2);
    context.fillStyle = "#df6235";
    context.fillRect(px + 6, py + 6, 5, 6);
    context.fillRect(px + 3, py + 7, 3, 2);
    context.fillRect(px + 11, py + 7, 3, 2);
    context.fillStyle = "#324d62";
    context.fillRect(px + 6, py + 12, 2, 3);
    context.fillRect(px + 9, py + 12, 2, 3);
  } else {
    // Horizontal four-legged animal with a head and visible tail.
    context.fillStyle = "rgba(34,54,38,.2)";
    context.fillRect(px + 2, py + 13, 12, 1);
    context.fillStyle = "#367a50";
    context.fillRect(px + 4, py + 6, 9, 6);
    context.fillRect(px + 11, py + 4, 4, 6);
    context.fillRect(px + 2, py + 7, 3, 2);
    context.fillStyle = "#8fbd63";
    context.fillRect(px + 6, py + 7, 5, 2);
    context.fillStyle = "#285c43";
    context.fillRect(px + 5, py + 12, 2, 3);
    context.fillRect(px + 11, py + 11, 2, 4);
    context.fillStyle = "#eef4d8";
    context.fillRect(px + 13, py + 6, 1, 1);
  }
}

function renderInspector() {
  const selected = selectedAgentId
    ? state.agents.filter(item => item.id === selectedAgentId)
    : state.agents.filter(item => item.type === selectedScope);
  const individual = selectedAgentId ? selected[0] : null;
  ui["all-humans-button"].classList.toggle("active", !selectedAgentId && selectedScope === "human");
  ui["all-animals-button"].classList.toggle("active", !selectedAgentId && selectedScope === "animal");
  if (!selected.length) {
    ui["agent-empty"].hidden = false;
    ui["agent-details"].hidden = true;
    return;
  }
  const view = aggregateAgents(selected);
  ui["agent-empty"].hidden = true;
  ui["agent-details"].hidden = false;
  ui["agent-name"].textContent = individual
    ? individual.id
    : selectedScope === "human" ? "Todos los humanos" : "Todos los animales";
  ui["agent-kind"].textContent = individual
    ? `${individual.type} · ${individual.alive ? "vivo" : "muerto"}`
    : `grupo · ${selected.filter(agent => agent.alive).length}/${selected.length} vivos`;
  ui["brain-view-label"].textContent = individual
    ? `${individual.id} · ${individual.alive ? "vivo" : "muerto"}`
    : `${selectedScope === "human" ? "Humanos" : "Animales"} · promedio de ${selected.length} brains`;
  const vitals = [
    ["Salud", view.health, false], ["Energía", view.energy, false],
    ["Hambre", view.hunger, true], ["Sed", view.thirst, true]
  ];
  ui.vitals.innerHTML = vitals.map(([name, value, inverse]) => `
    <div class="vital-row"><span>${name}</span><div class="vital-track"><div class="vital-fill ${inverse ? "warning" : ""}" style="width:${value * 100}%"></div></div><b>${format(value, 2)}</b></div>
  `).join("");
  ui["agent-action"].textContent = view.action;
  ui["agent-reward"].textContent = signed(view.reward, 3);
  ui["agent-loss"].textContent = view.loss == null ? "—" : format(view.loss, 4);
  ui["agent-mode"].textContent = individual
    ? `${individual.exploration_profile === "scout" ? "Explorador Horde" : "Perfil normal"} · ${view.exploration > 0 ? "explora ahora" : "decide el brain"} · ε ${format(view.epsilon, 2)}${individual.governor_override ? ` · gobernador cambió ${individual.brain_preferred_action} → ${individual.action}` : ""}${individual.survival_priority ? ` · prioridad ${individual.survival_priority}` : ""}`
    : `${view.scouts}/${selected.length} exploradores · ${format(view.exploration * 100, 0)}% explorando ahora · ε̄ ${format(view.epsilon, 2)} · gobernador interviene en ${format(view.governor_override * 100, 0)}%`;
  ui["epsilon-explanation"].textContent = `ε ${format(view.epsilon, 2)} significa cerca de ${format(view.epsilon * 100, 0)}% de probabilidad de probar una acción aleatoria. Es un rasgo individual persistente: perfiles normales 1–15%; exploradores Horde 50%.`;
  ui["agent-replay"].textContent = individual
    ? `Personal ${view.replay_size} · Horde ${view.horde_replay_size}`
    : `Horde ${view.horde_replay_size} · personal medio ${format(view.replay_size, 0)}`;
  ui["agent-updates"].textContent = individual ? `${view.training_steps} updates` : `${format(view.training_steps, 0)} updates medios`;

  drawBrainNetwork(view, individual ? individual.id : `all-${selectedScope}`);
  renderSpatialMemory(selected, individual);

  const qValues = view.q_values;
  const extent = Math.max(...qValues.map(Math.abs), .001);
  ui["q-values"].innerHTML = ACTIONS.map((action, index) => `
    <div class="bar-row ${action === view.action ? "chosen" : ""}"><span>${action}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.abs(qValues[index] || 0) / extent * 100}%"></div></div><b>${format(qValues[index] || 0, 3)}</b></div>
  `).join("");

  ui.observations.innerHTML = view.observation_labels.map((label, index) => `<div class="observation"><span>${label}</span><b>${format(view.observation[index] || 0, 3)}</b></div>`).join("");
  const components = Object.entries(view.reward_components);
  ui["reward-components"].innerHTML = components.length ? components.map(([name, value]) => `<span class="component ${value > 0 ? "positive" : value < 0 ? "negative" : ""}">${name} ${signed(value, 2)}</span>`).join("") : `<span class="component">Sin acción todavía</span>`;
}

function aggregateAgents(agents) {
  const first = agents[0];
  const mean = key => agents.reduce((sum, agent) => sum + Number(agent[key] || 0), 0) / agents.length;
  const meanNullable = key => {
    const values = agents.map(agent => agent[key]).filter(value => value != null);
    return values.length ? values.reduce((sum, value) => sum + Number(value), 0) / values.length : null;
  };
  const averageVectors = vectors => {
    const width = Math.max(0, ...vectors.map(vector => vector.length));
    return Array.from({length: width}, (_, index) => vectors.reduce((sum, vector) => sum + Number(vector[index] || 0), 0) / vectors.length);
  };
  const actionCounts = new Map();
  for (const agent of agents) actionCounts.set(agent.action, (actionCounts.get(agent.action) || 0) + 1);
  const action = [...actionCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "WAITING";
  const componentNames = new Set(agents.flatMap(agent => Object.keys(agent.reward_components || {})));
  const rewardComponents = Object.fromEntries([...componentNames].map(name => [name, agents.reduce((sum, agent) => sum + Number(agent.reward_components?.[name] || 0), 0) / agents.length]));
  const activationNames = new Set(agents.flatMap(agent => Object.keys(agent.brain_activations || {})));
  const activations = Object.fromEntries([...activationNames].map(name => [name, averageVectors(agents.map(agent => agent.brain_activations?.[name] || []))]));
  const layerNames = first.weight_statistics?.map(row => row.layer) || [];
  const weights = layerNames.map(layer => {
    const rows = agents.map(agent => agent.weight_statistics.find(row => row.layer === layer)).filter(Boolean);
    return {layer, mean_abs: rows.reduce((sum, row) => sum + row.mean_abs, 0) / rows.length, max_abs: rows.reduce((sum, row) => sum + row.max_abs, 0) / rows.length};
  });
  return {
    health: mean("health"), energy: mean("energy"), hunger: mean("hunger"), thirst: mean("thirst"),
    reward: mean("reward"), loss: meanNullable("loss"), epsilon: mean("epsilon"),
    exploration: agents.filter(agent => agent.exploration).length / agents.length,
    governor_override: agents.filter(agent => agent.governor_override).length / agents.length,
    scouts: agents.filter(agent => agent.exploration_profile === "scout").length,
    replay_size: mean("replay_size"), horde_replay_size: mean("horde_replay_size"),
    training_steps: mean("training_steps"), action,
    q_values: averageVectors(agents.map(agent => agent.q_values || [])),
    observation: averageVectors(agents.map(agent => agent.observation || [])),
    observation_labels: [...(first.need_labels || []), ...(first.spatial_labels || [])],
    brain_architecture: first.brain_architecture, brain_activations: activations,
    weight_statistics: weights, reward_components: rewardComponents,
  };
}

function renderSpatialMemory(agents, individual) {
  if (individual) {
    ui["spatial-memory"].innerHTML = ["food", "water"].map(resource => {
      const memory = individual.spatial_memory?.[resource] || {};
      const label = resource === "food" ? "Comida" : "Agua";
      const place = memory.position ? `(${memory.position.join(", ")})` : "no vista";
      const age = memory.age == null ? "—" : `${memory.age} ticks`;
      return `<div class="memory-card"><strong>${label}</strong><span>${place}</span><small>edad ${age} · confianza ${format((memory.confidence || 0) * 100, 0)}%</small></div>`;
    }).join("");
    return;
  }
  ui["spatial-memory"].innerHTML = ["food", "water"].map(resource => {
    const remembered = agents.filter(agent => agent.spatial_memory?.[resource]?.position).length;
    const confidence = agents.reduce((sum, agent) => sum + Number(agent.spatial_memory?.[resource]?.confidence || 0), 0) / agents.length;
    return `<div class="memory-card"><strong>${resource === "food" ? "Comida" : "Agua"}</strong><span>${remembered}/${agents.length} recuerdan</span><small>confianza media ${format(confidence * 100, 0)}%</small></div>`;
  }).join("");
}

function drawBrainNetwork(view, viewKey) {
  const canvas = ui["brain-network"];
  const context = canvas.getContext("2d");
  const architecture = view.brain_architecture || {};
  const activations = view.brain_activations || {};
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#eef1e9";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "rgba(255,255,255,.46)";
  context.fillRect(24, 52, 510, 420);
  context.fillStyle = "rgba(219,232,223,.72)";
  context.fillRect(570, 52, 300, 420);
  context.fillStyle = "rgba(232,224,239,.55)";
  context.fillRect(910, 52, 380, 420);
  context.font = "700 11px system-ui";
  context.fillStyle = "#79847d";
  context.fillText("1 · INTERPRETAR", 42, 78);
  context.fillText("2 · INTEGRAR", 590, 78);
  context.fillText("3 · DECIDIR", 930, 78);
  const layers = [
    {key: "need_inputs", label: `Necesidades · ${architecture.need_input_size || 0} entradas`, x: 75, y: 190, h: 180, color: "#dc6b35"},
    {key: "need_hidden", label: `Prioridad vital · ${architecture.hidden_sizes?.[0] || 0}`, x: 310, y: 190, h: 180, color: "#e28a3f"},
    {key: "spatial_inputs", label: `Mundo y memoria · ${architecture.spatial_input_size || 0}`, x: 75, y: 385, h: 120, color: "#367fa7"},
    {key: "spatial_hidden", label: `Comprensión espacial · ${architecture.hidden_sizes?.[1] || 0}`, x: 310, y: 385, h: 120, color: "#52a2bd"},
    {key: "fusion_hidden", label: `Fusión · ${architecture.hidden_sizes?.[2] || 0} neuronas`, x: 720, y: 278, h: 310, color: "#277657"},
    {key: "q_values", label: "Q-values · 8 acciones", x: 1035, y: 278, h: 310, color: "#78588f"},
  ];
  const positions = Object.fromEntries(layers.map(layer => [layer.key, nodePositions(layer, activations[layer.key] || [])]));
  for (const [from, to] of [["need_inputs", "need_hidden"], ["spatial_inputs", "spatial_hidden"], ["need_hidden", "fusion_hidden"], ["spatial_hidden", "fusion_hidden"], ["fusion_hidden", "q_values"]]) {
    for (const a of positions[from]) for (const b of positions[to]) {
      const signal = Math.min(1, (Math.abs(a.value) + Math.abs(b.value)) / 2);
      context.strokeStyle = `rgba(55, 105, 79, ${.035 + signal * .16})`;
      context.lineWidth = .65 + signal * 1.1;
      context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    }
  }
  const pulse = .9 + Math.sin(performance.now() / 170) * .1;
  for (const layer of layers) {
    context.fillStyle = "#526158"; context.font = "650 12px system-ui"; context.textAlign = "center";
    context.fillText(layer.label, layer.x, layer.y - layer.h / 2 - 18);
    positions[layer.key].forEach((node, index) => {
      const value = node.value;
      const strength = Math.min(1, Math.abs(value));
      context.globalAlpha = .3 + strength * .7;
      context.fillStyle = layer.key === "q_values" && ACTIONS[index] === view.action ? "#f2ae52" : layer.color;
      if (strength > .45) {
        context.shadowColor = layer.key === "q_values" ? "#d7a75e" : layer.color;
        context.shadowBlur = 7 + strength * 9 * pulse;
      }
      context.beginPath(); context.arc(node.x, node.y, 5 + strength * 4 * pulse, 0, Math.PI * 2); context.fill();
      context.shadowBlur = 0;
      context.globalAlpha = 1;
      context.strokeStyle = "rgba(255,255,255,.75)";
      context.lineWidth = 1;
      context.stroke();
      if (layer.key === "q_values") {
        context.fillStyle = ACTIONS[index] === view.action ? "#9a541f" : "#5f5664";
        context.font = ACTIONS[index] === view.action ? "750 11px system-ui" : "500 10px system-ui";
        context.textAlign = "left";
        context.fillText(`${ACTIONS[index]}  ${format(value, 3)}`, node.x + 18, node.y + 4);
        context.textAlign = "center";
      }
      context.globalAlpha = 1;
    });
  }
  context.textAlign = "start";
  const currentMeans = Object.fromEntries(view.weight_statistics.map(row => [row.layer, row.mean_abs]));
  const previous = previousWeightMeans.get(viewKey);
  const averageWeight = Object.values(currentMeans).reduce((sum, value) => sum + value, 0) / Math.max(1, Object.values(currentMeans).length);
  const previousAverage = previous ? Object.values(previous).reduce((sum, value) => sum + value, 0) / Math.max(1, Object.values(previous).length) : null;
  const delta = previousAverage == null ? null : averageWeight - previousAverage;
  previousWeightMeans.set(viewKey, currentMeans);
  ui["brain-network-caption"].textContent = `Los nodos brillantes están más activos. La acción resaltada es ${view.action}. Fuerza media de pesos ${format(averageWeight, 5)}${delta == null ? "" : ` · cambio desde la última lectura ${signed(delta, 7)}`}.`;
}

function nodePositions(layer, values) {
  const count = Math.max(1, Math.min(16, values.length));
  return Array.from({length: count}, (_, index) => {
    const sourceIndex = values.length <= count ? index : Math.round(index * (values.length - 1) / Math.max(1, count - 1));
    return {x: layer.x, y: layer.y - layer.h / 2 + (index + 1) * layer.h / (count + 1), value: Number(values[sourceIndex] || 0)};
  });
}

function renderEvents() {
  const events = (state.recent_events || []).slice(0, 24);
  ui["event-stream"].innerHTML = events.length ? events.map(event => `
    <div class="event"><span class="tick">#${event.tick}</span><strong>${event.agent_id}<br>${event.action}</strong><span class="${event.trained ? "trained" : ""}">${event.trained ? `↻ ${format(event.loss, 4)}` : signed(event.reward, 2)}</span></div>
  `).join("") : `<div class="empty-state" style="min-height:100px">Inicia o avanza un tick.</div>`;
}

function renderTrainingState() {
  const trained = state.agents.filter(agent => agent.trained).length;
  const flowItems = document.querySelectorAll("#learning-flow span");
  flowItems.forEach(item => item.classList.toggle("active", state.status === "running" && (item.textContent !== "Backward" || trained > 0)));
  if (state.status === "completed") {
    const reason = state.termination_reason === "human_extinction"
      ? `Run detenido en el tick ${state.tick}: murió el último humano.`
      : `Run terminado en el tick ${state.tick}.`;
    const champion = state.brb_promoted ? " Este run superó el campeón anterior y ahora es el nuevo BRB." : "";
    ui["training-message"].textContent = `${reason}${champion} Checkpoints y métricas guardados en ${state.result?.results_dir || "results/"}.`;
  } else if (state.tick === 0 && state.brb_source) {
    ui["training-message"].textContent = `Nuevo experimento desde BRB: ${state.agents.length} brains heredaron pesos de ${state.brb_source}. Adam, replay Horde, cuerpos y mundo comenzaron limpios.`;
  } else if (trained > 0) {
    ui["training-message"].textContent = `${trained} brains actualizaron sus pesos después de reunir todas las experiencias del tick en Horde. La loss proviene de backpropagation real.`;
  } else if (state.tick === 0 && state.continued_from) {
    const migration = state.architecture_migrations?.length
      ? ` ${state.architecture_migrations.length} brains fueron ensanchados conservando sus outputs iniciales.`
      : "";
    const rewardReset = state.learning_state_resets?.length
      ? ` Se limpiaron replay, Adam y target porque cambió la función de reward; se conservaron los perfiles epsilon individuales.`
      : "";
    ui["training-message"].textContent = `Nuevo ciclo con seed ${state.seed}: los ${state.agents.length} brains fueron cargados desde ${state.continued_from}.${migration}${rewardReset} El mundo físico comenzó de nuevo.`;
  } else {
    const humanHorde = state.horde?.human_replay_size || 0;
    const animalHorde = state.horde?.animal_replay_size || 0;
    ui["training-message"].textContent = `Horde comparte experiencias por especie: humanos ${humanHorde}, animales ${animalHorde}. Cada brain mantiene pesos propios y aprende de ese replay colectivo.`;
  }
}

function renderCompletionSummary() {
  const summary = state.learning_summary;
  ui["completion-summary"].hidden = !summary;
  if (!summary) {
    ui["comparison-chart-wrap"].hidden = true;
    return;
  }
  const extinct = summary.termination_reason === "human_extinction";
  ui["termination-label"].textContent = extinct ? `Extinción humana · tick ${summary.ticks_executed}` : `Límite alcanzado · tick ${summary.ticks_executed}`;
  ui["survivor-summary"].textContent = `Humanos: ${summary.initial_humans} → ${summary.final_humans} · Animales: ${summary.initial_animals} → ${summary.final_animals}`;
  const comparison = summary.comparison_to_previous_run;
  ui["run-comparison"].hidden = !comparison;
  if (comparison) {
    const change = comparison.last_human_tick_change;
    ui["run-comparison"].textContent = `Run anterior → actual: último humano ${comparison.previous_last_human_tick} → ${comparison.current_last_human_tick} ticks (${change >= 0 ? "+" : ""}${change}); supervivencia humana media ${format(comparison.previous_mean_human_survival, 1)} → ${format(comparison.current_mean_human_survival, 1)}.`;
  }
  drawRunComparison(comparison);
  ui["learning-numbers"].innerHTML = [["human", "Humanos"], ["animal", "Animales"]].map(([key, label]) => {
    const values = summary.by_type[key];
    return `<tr><td><strong>${label}</strong></td><td>${signed(values.first_20_percent_reward, 4)}</td><td>${signed(values.last_20_percent_reward, 4)}</td><td>${format(values.first_20_percent_loss, 4)}</td><td>${format(values.last_20_percent_loss, 4)}</td><td>${values.successful_drinks} (${values.brain_selected_drinks} brain)</td></tr>`;
  }).join("");
  ui["learned-explanation"].innerHTML = summary.what_they_learned.map(sentence => `<li>${sentence}</li>`).join("");
}

function drawRunComparison(comparison) {
  ui["comparison-chart-wrap"].hidden = !comparison;
  if (!comparison) return;
  const canvas = ui["comparison-chart"];
  const context = canvas.getContext("2d");
  const data = [
    {label: "Último humano", previous: comparison.previous_last_human_tick, current: comparison.current_last_human_tick},
    {label: "Supervivencia media", previous: comparison.previous_mean_human_survival, current: comparison.current_mean_human_survival}
  ];
  const maxValue = Math.max(1, ...data.flatMap(item => [item.previous, item.current]));
  const baseline = 210;
  const chartHeight = 150;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f4f3ed";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#17201c";
  context.font = "600 13px system-ui";
  context.fillText(`Run ${pad(comparison.previous_run_number)} vs Run ${pad(state.run_number)}`, 28, 28);
  context.font = "11px system-ui";
  context.fillStyle = "#6d7771";
  context.fillText("Run anterior", 500, 27);
  context.fillStyle = "#9aa39c";
  context.fillRect(480, 18, 12, 12);
  context.fillStyle = "#6d7771";
  context.fillText("Run actual", 610, 27);
  context.fillStyle = "#d97338";
  context.fillRect(590, 18, 12, 12);
  data.forEach((item, index) => {
    const groupX = 115 + index * 310;
    const previousHeight = item.previous / maxValue * chartHeight;
    const currentHeight = item.current / maxValue * chartHeight;
    context.fillStyle = "#9aa39c";
    context.fillRect(groupX, baseline - previousHeight, 62, previousHeight);
    context.fillStyle = "#d97338";
    context.fillRect(groupX + 72, baseline - currentHeight, 62, currentHeight);
    context.textAlign = "center";
    context.font = "600 11px system-ui";
    context.fillStyle = "#17201c";
    context.fillText(format(item.previous, 1), groupX + 31, baseline - previousHeight - 7);
    context.fillText(format(item.current, 1), groupX + 103, baseline - currentHeight - 7);
    context.font = "11px system-ui";
    context.fillStyle = "#6d7771";
    context.fillText(item.label, groupX + 67, baseline + 22);
    context.textAlign = "start";
  });
}

function showError(message) {
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  setTimeout(() => { ui.toast.hidden = true; }, 5000);
}
function statusText(status) {
  return ({paused: "PAUSADO", running: "ENTRENANDO", preparing: "PREPARANDO", finalizing: "GUARDANDO", completed: "COMPLETO", error: "ERROR"})[status] || status.toUpperCase();
}
function format(value, digits = 2) { return Number(value).toFixed(digits); }
function signed(value, digits = 2) { const number = Number(value); return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`; }
function pad(value) { return String(value).padStart(3, "0"); }

ui["play-button"].addEventListener("click", () => control("play"));
ui["pause-button"].addEventListener("click", () => control("pause"));
ui["step-button"].addEventListener("click", () => control("step"));
ui["next-run-button"].addEventListener("click", () => control("next_run"));
ui["all-humans-button"].addEventListener("click", () => {
  selectedAgentId = null;
  selectedScope = "human";
  renderWorld();
  renderInspector();
});
ui["all-animals-button"].addEventListener("click", () => {
  selectedAgentId = null;
  selectedScope = "animal";
  renderWorld();
  renderInspector();
});
ui["speed-input"].addEventListener("input", event => { ui["speed-label"].textContent = `${event.target.value} ticks/s`; });
ui["speed-input"].addEventListener("change", event => control("speed", Number(event.target.value)));
for (const id of ["human-count-input", "animal-count-input", "human-brain-input", "animal-brain-input"]) {
  ui[id].addEventListener("input", () => {
    configurationDirty = true;
    updateBuilderLabels();
  });
}
ui["use-brb-input"].addEventListener("change", () => {
  configurationDirty = true;
  applyBrbMinimums();
  renderExperimentBuilder();
});
ui["apply-config-button"].addEventListener("click", () => {
  const settings = {
    num_humans: Number(ui["human-count-input"].value),
    num_animals: Number(ui["animal-count-input"].value),
    human_brain_width: Number(ui["human-brain-input"].value),
    animal_brain_width: Number(ui["animal-brain-input"].value),
    use_brb: ui["use-brb-input"].checked
  };
  selectedAgentId = null;
  selectedScope = "human";
  configurationDirty = false;
  control("new_experiment", settings);
});
ui.world.addEventListener("click", event => {
  if (!state) return;
  const bounds = ui.world.getBoundingClientRect();
  const x = Math.floor((event.clientX - bounds.left) / bounds.width * state.grid.width);
  const y = Math.floor((event.clientY - bounds.top) / bounds.height * state.grid.height);
  const candidates = state.agents.filter(agent => agent.x === x && agent.y === y);
  const agent = candidates.find(candidate => candidate.alive) || candidates[0];
  if (agent) {
    selectedAgentId = agent.id;
    selectedScope = agent.type;
    renderWorld();
    renderInspector();
  }
});

poll();
