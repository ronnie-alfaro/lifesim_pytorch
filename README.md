# LifeSim v0.1

LifeSim es un simulador de vida artificial en cuadrícula y un laboratorio pequeño para estudiar redes neuronales con PyTorch. Cinco humanos y diez animales intentan sobrevivir buscando comida y agua. Cada individuo posee su propio brain, optimizer e historial personal; los pesos no se comparten, pero Horde permite aprender del replay colectivo de su especie.

La versión 0.1 prioriza un ciclo completo y observable:

```text
WORLD -> PERCEPTION -> BRAIN -> ACTION -> REWARD -> LEARNING -> CHECKPOINT -> NEXT RUN
```

No intenta modelar biología realista. Tampoco requiere Docker, servidores, cuentas ni procesos auxiliares.

## Requisitos e instalación

- Python 3.12 o posterior
- PyTorch, NumPy, Pandas y Matplotlib
- pytest para desarrollo

Con `venv` y pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Con `uv`:

```bash
uv sync --extra dev
```

## Ejecutar

### Laboratorio web interactivo

La forma recomendada de observar la simulación es:

```bash
python main.py --new --web --seed 42
```

Después abre `http://127.0.0.1:8765`. La simulación comienza pausada para no perder los primeros ticks. La interfaz permite:

- iniciar, pausar o avanzar exactamente un tick;
- cambiar la velocidad entre 1 y 30 ticks por segundo;
- crear un experimento con 1–30 humanos y 1–50 animales mediante sliders;
- ajustar la anchura de los brains humanos y animales entre 8 y 64 neuronas;
- iniciar un experimento desde el **Best Result Brain (BRB)** mediante un checkbox;
- recorrer una matriz Canvas de 60×40 campos, renderizada internamente a 960×640 para conservar pixel art nítido dentro del mismo espacio visual;
- distinguir humanos verticales naranjas, animales cuadrúpedos horizontales, plantas con frutos, agua animada visualmente por ondas y obstáculos de roca;
- ver por defecto estadísticas agregadas de todos los humanos o todos los animales;
- seleccionar un agente en el mundo para aislar únicamente sus datos;
- observar debajo del mundo un diagrama Brain v2 de 1320×520: dos ramas —necesidades y memoria espacial—, conexiones cuya intensidad sigue la señal, fusión, ocho Q-values etiquetados y acción elegida;
- seguir la fuerza media de sus pesos y el cambio desde la lectura anterior;
- distinguir exploración aleatoria de una acción seleccionada por el brain;
- ver crecer el replay personal y el Horde replay colectivo, junto con cada actualización real de pesos y su loss;
- terminar normalmente y generar los mismos checkpoints, CSV y gráficos.

Para visualizar un run que parte de conocimiento anterior:

```bash
python main.py --web \
  --resume checkpoints/experiment_001/run_001 \
  --seed 43
```

La web usa el servidor HTTP incluido en Python y JavaScript/CSS sin frameworks ni CDNs. No necesita Node, Docker ni servicios externos. El navegador consulta el estado del mismo `SimulationEngine.step()` utilizado por consola: las acciones visibles son exactamente las experiencias que entran al entrenamiento.

El run se detiene inmediatamente si muere el último humano. En ese tick se guardan normalmente los brains, métricas y gráficos. La consola y la web muestran entonces:

- humanos y animales al inicio y al final;
- reward medio del primer y último 20%;
- loss del primer y último 20%;
- supervivencia media y acción más frecuente;
- gráfica comparativa del tick del último humano y la supervivencia media frente al run anterior;
- una sección **¿Qué aprendieron?** que explica los cambios con palabras sencillas y evita afirmar aprendizaje cuando los números no lo demuestran.

Cuando aparece el resumen, el botón **↻ Siguiente ciclo** situado en la barra superior se activa. No hace falta cerrar ni reiniciar el servidor: LifeSim guarda el run, reconstruye todos los brains desde ese checkpoint, verifica que sus pesos iniciales sean exactamente los pesos finales anteriores, crea un mundo nuevo, incrementa la seed y comienza el siguiente run. El resumen siguiente incluirá automáticamente la comparación gráfica entre ambos ciclos.

El panel **Nuevo experimento** permite elegir población y capacidad neuronal antes del primer tick o después de finalizar un run. El control de anchura dimensiona los codificadores y la capa de fusión de Brain v2. Sin marcar BRB, **Crear experimento** comienza con brains nuevos; **Siguiente ciclo** conserva la población, arquitectura, pesos, optimizer y replay aprendidos en la cadena actual. Los controles quedan bloqueados durante un run para evitar descartar entrenamiento accidentalmente. Cada agente continúa teniendo un modelo, optimizer y replay independientes, por lo que aumentar población y anchura incrementa el uso de CPU y memoria.

### Best Result Brain (BRB)

Al terminar cada run compatible, LifeSim compara el rendimiento del grupo humano con el campeón guardado. El contrato de puntuación BRB v2 prioriza, en este orden: completar el límite de ticks con al menos un humano vivo, proporción de humanos que completaron el ciclo, número absoluto de supervivientes, mediana individual, supervivencia media, bebidas elegidas por el brain, comidas y, finalmente, menos prioridades vitales ignoradas. Así, un run largo que termina en extinción no desplaza a uno que realmente alcanza el horizonte experimental. Los runs interrumpidos tampoco pueden ser campeones.

El baseline incluido actualmente es **experimento 033, run 001, seed 42**: alcanzó 5000 ticks con 2/5 humanos y 2/10 animales vivos. Sus pesos y arquitecturas son el BRB disponible para iniciar experimentos nuevos. Un resultado compatible solo lo reemplaza si su vector de supervivencia es estrictamente mejor; cambiar reward o versión de arquitectura inicia un contrato de aprendizaje distinto y reinicia Adam, target y replay cuando corresponde.

Si el nuevo resultado es estrictamente mejor, se crea una copia inmutable de sus pesos en:

```text
checkpoints/best_result_brain/
  registry.json
  champions/experiment_027_run_050/
    human_001.pt ... animal_010.pt
```

Al marcar **Usar Best Result Brain (BRB)**, el nuevo experimento hereda esos pesos, pero reinicia mundo, estado físico, optimizer Adam, replay personal y replay Horde. Esto permite variar la seed, la población y la capacidad sin confundir la comparación con memoria de entrenamiento anterior. Si se piden más agentes que los guardados, los mejores brains de cada especie se usan cíclicamente como padres; cada copia vuelve a ser un modelo PyTorch independiente y puede divergir al entrenar. La arquitectura puede mantenerse o ensancharse preservando los outputs iniciales, pero no encogerse por debajo del campeón; la web ajusta automáticamente el mínimo de los sliders.

### Modo consola

Nuevo experimento reproducible (5,000 ticks por defecto):

```bash
python main.py --new --seed 42
```

Para una prueba breve y representación ASCII opcional:

```bash
python main.py --new --seed 42 --ticks 200 --status-every 25 --render-every 100
```

Continuar el experimento usando exactamente los pesos finales del run anterior:

```bash
python main.py --resume checkpoints/experiment_001/run_001 --seed 43
```

El resume reconstruye cada arquitectura Brain v2, carga pesos, target network, estado de Adam, replay buffer y contadores de exploración/entrenamiento. Después reinicia posición, salud, energía, hambre, sed y el mapa recordado porque el mundo es nuevo. Lo aprendido para usar las necesidades y orientarse sí permanece en los pesos. Brain v2 puede ensanchar sus capas conservando numéricamente sus outputs iniciales; solo en esa migración se reinicia Adam porque sus tensores ya no tienen las mismas dimensiones.

Los checkpoints Brain v1 son estructuralmente incompatibles con las ramas de Brain v2. El cargador los rechaza con un error explícito: para comenzar esta etapa usa `--new`. A partir de ese primer run v2, **Siguiente ciclo** y `--resume` continúan normalmente.

### Entrenamiento continuo solo en texto

Para iniciar un experimento y encadenar runs indefinidamente:

```bash
python main.py --new --continuous --text-only --seed 42 --status-every 100
```

Cuando muere el último humano —o se alcanza el límite de ticks— se guarda el checkpoint y comienza automáticamente otro mundo con los brains aprendidos. El proceso continúa hasta pulsar `Ctrl+C`. La salida compacta muestra el run, tick, supervivientes, bebidas humanas, epsilon y loss. Al cerrar cada ciclo muestra el tick final, bebidas elegidas por el brain, muertes asociadas a sed crítica y la comparación con el run anterior.

Para continuar una cadena existente:

```bash
python main.py \
  --resume checkpoints/experiment_001/run_007 \
  --continuous --text-only --seed 49
```

`--text-only` evita el renderer ASCII y la generación de PNG para que los ciclos sean más rápidos. Los checkpoints, `agents.csv`, `summary.csv` y `run_summary.json` sí se conservan. Si se pulsa `Ctrl+C` durante un run que ya avanzó, LifeSim intenta guardar también ese run parcial con la razón `user_interrupt`.

`--debug` imprime los componentes que formaron cada reward. Es verboso. `--ticks`, `--status-every` y `--render-every` son overrides de ejecución útiles para validar. En modo continuo, `--ticks` define el límite de cada ciclo individual, no el límite de toda la cadena.

## Arquitectura

```text
agents/       estado individual, percepción, AgentBrain y clases Human/Animal
world/        cuadrícula, recursos, acciones y renderer ASCII desacoplado
learning/     reward observable, replay buffer y entrenamiento DQN-style
simulation/   engine, métricas y gestión de experimentos/runs
persistence/  checkpoints reconstruibles e integrity hashes
analysis/     seis gráficos del run y comparaciones entre runs
web/          servidor local y laboratorio interactivo
tests/        comportamiento, aprendizaje y round-trip de checkpoints
```

### Runtime y concurrencia

LifeSim es local-first y autocontenido. `main.py` crea un único `SimulationEngine`; el modo web añade un `ThreadingHTTPServer` de la biblioteca estándar y un hilo controlador. Un `RLock` protege el cambio de estado entre los requests HTTP y el loop de ticks. No existe un segundo simulador en JavaScript: Canvas solo dibuja snapshots JSON del mismo engine que entrena los brains.

La API local deliberadamente pequeña es:

```text
GET  /api/health   estado básico del controlador
GET  /api/state    snapshot del mundo, agentes, activaciones y métricas
POST /api/control  play, pause, step, speed, next_run o new_experiment
```

Los assets HTML, CSS y JavaScript se sirven con `Cache-Control: no-store`, de modo que basta refrescar el navegador durante desarrollo. No hay WebSocket: mientras corre, el cliente consulta `/api/state` aproximadamente cada 160 ms. La velocidad visual controla cuántos ticks ejecuta el hilo por segundo; no altera las ecuaciones metabólicas ni el contenido de cada experiencia.

Toda configuración experimental vive en `config.py`. Brain v2 se construye dinámicamente con tres tamaños: `[codificador_necesidades, codificador_espacial, fusión]`. El humano usa por defecto supervivencia `15 → 16`, espacio/memoria `18 → 32`, fusión `48 → 32` y salida `32 → 8`. El animal usa `15 → 12`, `16 → 24`, fusión `36 → 24` y salida `24 → 8`. Se pueden cambiar esos tamaños, learning rate, batch size, gamma, frecuencia de actualización de la target network y capacidad del replay buffer sin tocar el modelo.

## Percepción y decisiones

Las observaciones son tensores pequeños y documentados en `agents/human.py` y `agents/animal.py`. Los primeros quince valores forman la rama de supervivencia: hambre, sed, falta de energía y salud, cuatro banderas de prioridad, riesgo progresivo de hambre/sed/agotamiento, daño activo, daño reciente, margen de vida estimado y urgencia vital combinada. El riesgo comienza a crecer desde el 50%, antes de que la salud empiece a caer. El resto forma la rama espacial: memoria de comida y agua, confianza y edad del recuerdo, obstáculos cardinales, posición y recursos al alcance. Los humanos añaden distancia a otros humanos y animales.

La visión de obstáculos es local (`vision_radius = 6`), pero comida y agua producen un rastro de largo alcance (`resource_sense_radius = 100`). El agente conserva un objetivo espacial y, cuando una necesidad se vuelve prioritaria, mantiene ese destino mientras el recurso siga existiendo. Si otro agente consume la comida, corrige el recuerdo y busca otro objetivo. Esta separación evita que un mapa de 60×40 convierta la búsqueda inicial en azar puro.

La red produce ocho Q-values, uno por acción: mover en cuatro direcciones, comer, beber, descansar o esperar. Epsilon-greedy decide entre exploración aleatoria y `argmax` de esos Q-values. **Epsilon (ε) es la probabilidad de ignorar temporalmente la decisión favorita del brain y probar una acción aleatoria.** Ya no existe un epsilon global que comience en 100%: es un rasgo individual persistente. Aproximadamente el 90% de cada especie recibe un perfil normal entre `0.01` y `0.15`; una minoría exploradora estable —10%, al menos un individuo— usa `0.50`. Así la mayoría explota lo aprendido y algunos exploradores continúan produciendo experiencias nuevas. En estado seguro el gobernador solo elimina acciones físicamente inválidas; al aparecer una prioridad vital limita temporalmente el conjunto a rutas que preservan la supervivencia. La elección entre las opciones permitidas sigue saliendo de los Q-values del brain.

### Aprendizaje colectivo Horde

Cada humano conserva un brain y optimizer propios, pero entrena muestreando un replay compartido por todos los humanos. Los animales hacen lo mismo en otro replay separado. Cada tick tiene dos fases sincronizadas: primero todos actúan y depositan sus transiciones; después todos los brains hacen su actualización con el Horde ya completo para ese tick. Esto elimina la ventaja artificial del último agente procesado. Si un humano descubre cómo beber, esa transición queda disponible para que todos los brains humanos la estudien; no es necesario que cada individuo descubra el mismo evento por accidente. El checkpoint guarda `horde_replay.pt` con hash de integridad y lo recupera en el ciclo siguiente. El replay personal también se conserva para observabilidad.

Esta primera implementación es **Horde-inspired collective replay**: toma la idea de aprender muchas predicciones o políticas a partir de experiencia compartida, pero todavía no implementa el Horde académico completo basado en múltiples General Value Functions o “demons”. Esa extensión puede añadirse después para que el brain aprenda predicciones separadas como “probabilidad de encontrar agua” o “riesgo de morir en N ticks”.

Un agente puede comer o beber desde su celda o desde una celda cardinal adyacente. Cuando tiene una necesidad relevante recibe una señal pequeña por acercarse al recurso urgente y una señal negativa por alejarse; el reward grande continúa reservado para comer o beber realmente. Así se mantiene la decisión en el brain, pero deja de depender de una coincidencia extremadamente rara de posición y acción.

El agua se genera en varios clusters grandes y representa fuentes permanentes: beber no elimina una casilla. La comida también aparece en clusters, sí se consume al comer y vuelve a crecer gradualmente junto a plantas existentes hasta `max_food`. Los centros se distribuyen por máxima separación y cada zona de comida tiene agua cercana, formando hábitats alcanzables en lugar de dejar desiertos aleatorios. Cantidades, número de clusters y probabilidad de rebrote viven en `config.py`.

### Reward proporcional a la necesidad

Comer, beber y descansar no entregan un premio fijo. Si hambre, sed o falta de energía están por encima de `need_action_threshold`, el reward es proporcional a la necesidad previa. Por ejemplo, beber con sed `0.80` aporta aproximadamente `+0.80`, mientras beber con sed `0.01` es innecesario y no cuenta como bebida exitosa.

Las acciones innecesarias acumulan una penalización independiente por tipo: `-0.10`, `-0.20`, `-0.30` y así sucesivamente hasta `-1.00`. La racha solo se reinicia cuando esa misma acción vuelve a satisfacer una necesidad real. Esto evita obtener reward infinito bebiendo cada tick de una fuente permanente. El discount factor es `gamma = 0.99` para que consecuencias tardías como morir de hambre influyan más en decisiones anteriores.

Reward v5 organiza hambre y sed en tres zonas. Desde 25% comienza la planificación protegida; el objetivo seguro es permanecer en 50% o menos; 70% abre la zona de peligro. Cada tick por encima del 50% tiene un costo cuadrático que llega a `-0.30` al alcanzar 70% y puede crecer hasta `-0.80`. Volver realmente a la zona segura entrega `+0.25`, además del reward proporcional de comer o beber. Como el mapa duplicó sus dimensiones originales pero los agentes siguen avanzando una celda por tick, hambre y sed aumentan `0.005` por tick para conservar el presupuesto metabólico por distancia.

Reward v5 añade un **gobernador de supervivencia** observable. El brain continúa produciendo los ocho Q-values, pero una máscara elimina acciones físicamente imposibles y decisiones incompatibles con supervivencia: no puede elegir `DRINK` sin agua, `EAT` sin comida, atravesar obstáculos ni alejarse de un recurso prioritario recordado. Cerca de recursos el brain elige entre las acciones seguras disponibles; durante exploración, epsilon también muestrea solamente acciones permitidas. La experiencia guarda la máscara del siguiente estado y el target DQN excluye los Q-values imposibles antes de calcular su máximo. La web muestra cuándo el gobernador cambió la preferencia original del brain.

`WAIT` y las acciones que no atienden una necesidad prioritaria reciben `ignored_survival_priority`. Un movimiento de búsqueda sigue permitido cuando el agente no conoce ningún recurso. Si recuerda comida o agua, acercarse recibe un reward que aumenta con la urgencia; alejarse o vagar recibe tanto progreso espacial negativo como penalización por ignorar la prioridad. El cálculo solo usa recursos visibles o recordados, nunca información oculta del mundo.

Los checkpoints guardan `reward_version`. Al cargar por primera vez un checkpoint creado con una función de reward antigua, LifeSim conserva los pesos del brain, pero limpia replay, Adam y target network porque estaban asociados al objetivo anterior. Los perfiles epsilon individuales se mantienen dentro del rango Horde. Los checkpoints posteriores con la misma versión continúan normalmente sin otro reinicio.

El replay sampling reserva inicialmente un 25% del batch para experiencias con reward positivo cuando existen. Esto permite volver a estudiar acciones escasas como beber exitosamente sin programar la decisión dentro del entorno. Los CSV y el resumen separan bebidas elegidas por el brain de bebidas ocurridas durante exploración aleatoria.

El aprendizaje puede seguirse directamente en `learning/trainer.py`:

```text
perception tensor
  -> forward and Q-values
  -> selected action Q-value
  -> replay sample
  -> bootstrapped target
  -> SmoothL1 loss
  -> zero_grad
  -> backward
  -> optimizer.step (weights change here)
```

El baseline DQN-style usa una target network independiente que se sincroniza periódicamente para estabilizar el objetivo. El forward de decisiones sigue ocurriendo en el brain individual; la target network se usa únicamente al calcular el valor futuro del batch.

Técnicamente, para cada muestra el trainer calcula `Q(s, a)` con `gather`, construye `reward + gamma * max(Q_target(s')) * (1 - done)` dentro de `torch.no_grad()`, excluye con una máscara las acciones imposibles del siguiente estado y minimiza `SmoothL1Loss`. Después ejecuta `zero_grad()`, `backward()`, clipping de gradiente a 10 y `optimizer.step()`. Cada 100 actualizaciones copia los pesos a la target network. Los sanity checks detienen el run ante loss o pesos no finitos.

### Límite conocido de ecología v0.1

El comportamiento neuronal y la capacidad ecológica deben medirse por separado. Con `hunger_per_tick = 0.005` y una comida que reduce hambre en `0.50`, cada agente necesita aproximadamente una comida cada 100 ticks. La población inicial de 15 agentes demanda cerca de `0.15` unidades por tick, mientras el rebrote actual intenta crear como máximo una unidad con probabilidad `0.08` por tick. Por eso el run 033 demuestra supervivencia individual y sirve como BRB, pero también muestra un cuello de botella: la reserva alimentaria se estabiliza solo después de que disminuye la población. La siguiente iteración debe hacer crecer comida por cluster o escalar la producción con la población antes de usar supervivencia grupal como prueba concluyente del aprendizaje.

## Persistencia y resultados

Cada run produce:

```text
checkpoints/experiment_001/run_001/
  human_001.pt ... animal_010.pt
  metadata.json

results/experiment_001/run_001/
  agents.csv
  summary.csv
  run_summary.json
  reward.png
  loss.png
  survival.png
  actions.png
  average_survival.png
  reward_progression.png
```

Los `.pt` contienen arquitectura, pesos, optimizer, identidad y estadísticas. `metadata.json` conserva seed, configuración, fuente del resume y hashes iniciales/finales. Los CSV registran estado individual y agregados por tick.

Los gráficos muestran recompensa acumulada individual, loss, agentes vivos, distribución de acciones, supervivencia media y progresión suavizada del reward. Desde el segundo run también aparecen `results/experiment_001/comparisons/reward_by_run.png` y `survival_by_run.png`.

El resumen first/last 20% es evidencia descriptiva para inspección; por sí solo no demuestra aprendizaje significativo.

## Tests

```bash
pytest
```

La suite actual contiene 55 tests y cubre creación del mundo y agentes, ramas y activaciones de Brain v2, perfiles epsilon individuales, Horde replay por especie, prioridad de necesidades, memoria espacial, movimiento, comida, bebida, backpropagation con cambio de pesos, checkpoint y reproducción exacta de outputs, selección BRB, rechazo claro de Brain v1 y sanity checks de shapes y límites.
