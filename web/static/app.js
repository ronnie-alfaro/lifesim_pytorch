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
  "architecture-preview", "builder-help",
  "all-humans-button", "all-animals-button",
  "agent-name", "agent-kind", "agent-empty", "agent-details", "vitals", "agent-action",
  "agent-reward", "agent-loss", "agent-mode", "agent-replay", "agent-updates", "epsilon-explanation",
  "brain-network", "brain-network-caption", "spatial-memory",
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
  if (!configurationDirty) {
    ui["human-count-input"].value = config.num_humans;
    ui["animal-count-input"].value = config.num_animals;
    ui["human-brain-input"].value = config.human_hidden_sizes.at(-1);
    ui["animal-brain-input"].value = config.animal_hidden_sizes.at(-1);
  }
  updateBuilderLabels();
  ui["apply-config-button"].disabled = !state.can_configure_experiment;
  ui["builder-help"].textContent = state.can_configure_experiment
    ? "Crea un experimento independiente con brains nuevos. Más agentes y neuronas requieren más CPU."
    : "Bloqueado durante el run. Termina el ciclo para crear otro experimento; Siguiente ciclo conserva los brains actuales.";
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
  ui["architecture-preview"].textContent = `Humano: necesidades 8→${Math.max(8, humanWidth / 2)}, espacio 18→${humanWidth}, fusión ${humanWidth}→8 · Animal: espacio 16→${animalWidth} · ${humans + animals} brains v2`;
}

function renderWorld() {
  const {width, height, food, water, obstacles} = state.grid;
  const canvas = ui.world;
  const cellSize = 12;
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

  context.fillStyle = "#e8e5d8";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) drawMatrixCell(context, matrix[y][x], x, y, cellSize);
  }
  for (const agent of state.agents) drawPixelAgent(context, agent, cellSize, agent.id === selectedAgentId);
}

function drawMatrixCell(context, value, x, y, size) {
  const px = x * size;
  const py = y * size;
  context.strokeStyle = "rgba(55, 67, 58, .07)";
  context.strokeRect(px + .5, py + .5, size - 1, size - 1);
  if (value === 1) {
    context.fillStyle = "#58605b";
    context.fillRect(px + 1, py + 1, size - 2, size - 2);
    context.fillStyle = "#707a73";
    context.fillRect(px + 2, py + 2, size - 5, 2);
  } else if (value === 2) {
    context.fillStyle = "#78943e";
    context.fillRect(px + 3, py + 3, size - 6, size - 5);
    context.fillStyle = "#a8bd62";
    context.fillRect(px + 4, py + 2, Math.max(2, size - 8), 3);
  } else if (value === 3) {
    context.fillStyle = "#397fa5";
    context.fillRect(px + 1, py + 1, size - 2, size - 2);
    context.fillStyle = "#76b8d3";
    context.fillRect(px + 2, py + 3, size - 5, 2);
    context.fillRect(px + 5, py + 7, size - 7, 2);
  }
}

function drawPixelAgent(context, agent, size, selected) {
  const px = agent.x * size;
  const py = agent.y * size;
  if (selected) {
    context.strokeStyle = "#fffdf4";
    context.lineWidth = 2;
    context.strokeRect(px + 1, py + 1, size - 2, size - 2);
    context.strokeStyle = "#17201c";
    context.lineWidth = 1;
    context.strokeRect(px + .5, py + .5, size - 1, size - 1);
  }
  if (!agent.alive) {
    context.strokeStyle = "rgba(60, 60, 57, .55)";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(px + 3, py + 3);
    context.lineTo(px + size - 3, py + size - 3);
    context.moveTo(px + size - 3, py + 3);
    context.lineTo(px + 3, py + size - 3);
    context.stroke();
    return;
  }
  if (agent.type === "human") {
    // A vertical pixel person: square head over a narrow body (o above -).
    context.fillStyle = "#f2ae52";
    context.fillRect(px + 4, py + 1, 4, 4);
    context.fillStyle = "#d76530";
    context.fillRect(px + 5, py + 5, 2, 6);
    context.fillRect(px + 3, py + 6, 6, 2);
  } else {
    // A horizontal pixel animal: head followed by a pointed body (o<).
    context.fillStyle = "#9ac45d";
    context.fillRect(px + 1, py + 4, 4, 4);
    context.fillStyle = "#236b50";
    context.fillRect(px + 5, py + 5, 5, 3);
    context.fillRect(px + 8, py + 3, 2, 2);
    context.fillRect(px + 8, py + 8, 2, 2);
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
    ? `${individual.exploration_profile === "scout" ? "Explorador Horde" : "Perfil normal"} · ${view.exploration > 0 ? "explora ahora" : "decide el brain"} · ε ${format(view.epsilon, 2)}`
    : `${view.scouts}/${selected.length} exploradores · ${format(view.exploration * 100, 0)}% explorando ahora · ε̄ ${format(view.epsilon, 2)}`;
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
  context.fillStyle = "#f4f3ed";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const layers = [
    {key: "need_inputs", label: `Necesidades ${architecture.need_input_size || 0}`, x: 55, y: 75, h: 105, color: "#d97338"},
    {key: "need_hidden", label: `Rama necesidad ${architecture.hidden_sizes?.[0] || 0}`, x: 185, y: 75, h: 105, color: "#d97338"},
    {key: "spatial_inputs", label: `Espacio/memoria ${architecture.spatial_input_size || 0}`, x: 55, y: 220, h: 125, color: "#397fa5"},
    {key: "spatial_hidden", label: `Rama espacial ${architecture.hidden_sizes?.[1] || 0}`, x: 185, y: 220, h: 125, color: "#397fa5"},
    {key: "fusion_hidden", label: `Fusión ${architecture.hidden_sizes?.[2] || 0}`, x: 355, y: 150, h: 155, color: "#236b50"},
    {key: "q_values", label: "8 acciones", x: 545, y: 150, h: 155, color: "#78588f"},
  ];
  const positions = Object.fromEntries(layers.map(layer => [layer.key, nodePositions(layer, activations[layer.key] || [])]));
  for (const [from, to] of [["need_inputs", "need_hidden"], ["spatial_inputs", "spatial_hidden"], ["need_hidden", "fusion_hidden"], ["spatial_hidden", "fusion_hidden"], ["fusion_hidden", "q_values"]]) {
    context.strokeStyle = "rgba(70, 79, 73, .13)";
    context.lineWidth = 1;
    for (const a of positions[from]) for (const b of positions[to]) {
      context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    }
  }
  for (const layer of layers) {
    context.fillStyle = "#667069"; context.font = "600 10px system-ui"; context.textAlign = "center";
    context.fillText(layer.label, layer.x, layer.y - layer.h / 2 - 16);
    positions[layer.key].forEach((node, index) => {
      const value = node.value;
      const strength = Math.min(1, Math.abs(value));
      context.globalAlpha = .25 + strength * .75;
      context.fillStyle = layer.key === "q_values" && ACTIONS[index] === view.action ? "#f2ae52" : layer.color;
      context.beginPath(); context.arc(node.x, node.y, 4 + strength * 3, 0, Math.PI * 2); context.fill();
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
  const count = Math.max(1, Math.min(10, values.length));
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
    ui["training-message"].textContent = `${reason} Checkpoints y métricas guardados en ${state.result?.results_dir || "results/"}.`;
  } else if (trained > 0) {
    ui["training-message"].textContent = `${trained} agentes actualizaron sus pesos en este tick. La loss mostrada proviene de backpropagation real.`;
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
ui["apply-config-button"].addEventListener("click", () => {
  const settings = {
    num_humans: Number(ui["human-count-input"].value),
    num_animals: Number(ui["animal-count-input"].value),
    human_brain_width: Number(ui["human-brain-input"].value),
    animal_brain_width: Number(ui["animal-brain-input"].value)
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
