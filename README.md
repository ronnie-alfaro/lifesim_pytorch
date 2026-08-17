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
- recorrer una matriz Canvas de 60×40 campos dentro del mismo espacio visual;
- distinguir humanos verticales naranjas, animales horizontales verdes, comida, agua y obstáculos;
- ver por defecto estadísticas agregadas de todos los humanos o todos los animales;
- seleccionar un agente en el mundo para aislar únicamente sus datos;
- observar Brain v2 como dos ramas —necesidades y memoria espacial—, sus activaciones, fusión, ocho Q-values y acción elegida;
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

El panel **Nuevo experimento** permite elegir población y capacidad neuronal antes del primer tick o después de finalizar un run. El control de anchura dimensiona los codificadores y la capa de fusión de Brain v2. **Crear experimento** comienza con brains nuevos; **Siguiente ciclo** conserva la población, arquitectura y pesos aprendidos. Los controles quedan bloqueados durante un run para evitar descartar entrenamiento accidentalmente. Cada agente continúa teniendo un modelo, optimizer y replay independientes, por lo que aumentar población y anchura incrementa el uso de CPU y memoria.

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

Toda configuración experimental vive en `config.py`. Brain v2 se construye dinámicamente con tres tamaños: `[codificador_necesidades, codificador_espacial, fusión]`. El humano usa por defecto necesidades `8 → 16`, espacio/memoria `18 → 32`, fusión `48 → 32` y salida `32 → 8`. El animal usa `8 → 12`, `16 → 24`, fusión `36 → 24` y salida `24 → 8`. Se pueden cambiar esos tamaños, learning rate, batch size, gamma, frecuencia de actualización de la target network y capacidad del replay buffer sin tocar el modelo.

## Percepción y decisiones

Las observaciones son tensores pequeños y documentados en `agents/human.py` y `agents/animal.py`. Los primeros ocho valores forman la rama de necesidades: hambre, sed, falta de energía, falta de salud y cuatro banderas de prioridad que se activan al llegar al 50%. El resto forma la rama espacial: memoria de comida y agua, confianza y edad del recuerdo, obstáculos cardinales, posición y recursos al alcance. Los humanos añaden distancia a otros humanos y animales.

La visión es local (`vision_radius = 6`). El agente solo descubre un recurso al verlo y conserva su última posición conocida con una confianza que disminuye durante 200 ticks. Si vuelve al lugar y el recurso ya no está, corrige el recuerdo. Esto es memoria explícita e interpretable; no es una regla de movimiento: el brain sigue decidiendo si usa esa información.

La red produce ocho Q-values, uno por acción: mover en cuatro direcciones, comer, beber, descansar o esperar. Epsilon-greedy decide entre exploración aleatoria y `argmax` de esos Q-values. **Epsilon (ε) es la probabilidad de ignorar temporalmente la decisión favorita del brain y probar una acción aleatoria.** Ya no existe un epsilon global que comience en 100%: es un rasgo individual persistente. Aproximadamente el 90% de cada especie recibe un perfil normal entre `0.01` y `0.15`; una minoría exploradora estable —10%, al menos un individuo— usa `0.50`. Así la mayoría explota lo aprendido y algunos exploradores continúan produciendo experiencias nuevas. No existe una regla programada como “si tiene hambre, ir a comida”.

### Aprendizaje colectivo Horde

Cada humano conserva un brain y optimizer propios, pero entrena muestreando un replay compartido por todos los humanos. Los animales hacen lo mismo en otro replay separado. Por eso, si un humano descubre cómo beber, esa transición queda disponible para que todos los brains humanos la estudien; no es necesario que cada individuo descubra el mismo evento por accidente. El checkpoint guarda `horde_replay.pt` con hash de integridad y lo recupera en el ciclo siguiente. El replay personal también se conserva para observabilidad.

Esta primera implementación es **Horde-inspired collective replay**: toma la idea de aprender muchas predicciones o políticas a partir de experiencia compartida, pero todavía no implementa el Horde académico completo basado en múltiples General Value Functions o “demons”. Esa extensión puede añadirse después para que el brain aprenda predicciones separadas como “probabilidad de encontrar agua” o “riesgo de morir en N ticks”.

Un agente puede comer o beber desde su celda o desde una celda cardinal adyacente. Cuando tiene una necesidad relevante recibe una señal pequeña por acercarse al recurso urgente y una señal negativa por alejarse; el reward grande continúa reservado para comer o beber realmente. Así se mantiene la decisión en el brain, pero deja de depender de una coincidencia extremadamente rara de posición y acción.

El agua se genera en varios clusters grandes y representa fuentes permanentes: beber no elimina una casilla. La comida también aparece en clusters, sí se consume al comer y vuelve a crecer gradualmente junto a plantas existentes hasta `max_food`. Cantidades, número de clusters y probabilidad de rebrote viven en `config.py`.

### Reward proporcional a la necesidad

Comer, beber y descansar no entregan un premio fijo. Si hambre, sed o falta de energía están por encima de `need_action_threshold`, el reward es proporcional a la necesidad previa. Por ejemplo, beber con sed `0.80` aporta aproximadamente `+0.80`, mientras beber con sed `0.01` es innecesario y no cuenta como bebida exitosa.

Las acciones innecesarias acumulan una penalización independiente por tipo: `-0.10`, `-0.20`, `-0.30` y así sucesivamente hasta `-1.00`. La racha solo se reinicia cuando esa misma acción vuelve a satisfacer una necesidad real. Esto evita obtener reward infinito bebiendo cada tick de una fuente permanente. El discount factor es `gamma = 0.99` para que consecuencias tardías como morir de hambre influyan más en decisiones anteriores.

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

La suite cubre creación del mundo y agentes, ramas y activaciones de Brain v2, perfiles epsilon individuales, Horde replay por especie, prioridad de necesidades, memoria espacial, movimiento, comida, bebida, backpropagation con cambio de pesos, checkpoint y reproducción exacta de outputs, rechazo claro de Brain v1 y sanity checks de shapes y límites.
