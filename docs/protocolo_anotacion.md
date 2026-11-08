# Protocolo de anotación de momentos decisivos (B8)

Las anotaciones son el **ground truth** con el que se mide el detector de momentos y, a partir de él, el valor en dinero (B10/B12). Si cada quien anota con un criterio distinto, las métricas no significan nada. Por eso: **lean esto completo antes de anotar la primera partida** y anoten las 15 compartidas sin hablar entre ustedes de lo que marcaron.

## Cómo abrir la herramienta

```
python src/select_annotation.py      # solo la primera vez (crea la selección y extrae los datos)
streamlit run src/annotate_app.py    # abre la herramienta en el navegador
```

En la barra lateral elijan su letra (A = compañero, B = Melisa) y una partida. Lo guardado queda en `annotations/A.csv` o `annotations/B.csv`, que **sí van a git**.

La herramienta muestra solo el **log de eventos** (kills, objetivos, torres, marcador), sin la curva del modelo ni la gráfica de oro. Es a propósito: si vieran la curva, marcarían lo que el modelo ya marca y la evaluación quedaría sesgada.

## Qué es un momento decisivo

> **Un momento decisivo es una jugada concreta, de a lo sumo ~2 minutos, después de la cual cambió claramente qué equipo era favorito, o se aseguró una victoria que antes no estaba asegurada.**

La pregunta para cada candidato es: *"Si un coach solo tuviera tiempo de ver 5 clips de esta partida, ¿este sería uno?"*

**Sí cuenta** (ejemplos típicos):
- **Pelea** en la que un equipo saca ventaja grande: 3 o más kills de diferencia, o un ace.
- **Objetivo mayor** que cambia la partida: barón, dragón ancestral (elder), alma de dragón, sobre todo si se consigue gracias a una pelea.
- **Pick** de un jugador clave justo antes de un objetivo, que deja al rival sin poder disputarlo.
- **Estructura** que abre la base: inhibidor, torres de base. Cuenta si fue resultado de una jugada, no si cayó sola al final.
- **Error / throw:** el equipo que iba ganando pierde una pelea o un objetivo y la partida se da vuelta.

**No cuenta:**
- Kills sueltas (1 contra 1) sin consecuencia.
- Placas o torres exteriores que caen solas por presión normal de línea.
- Objetivos tomados sin disputa que no cambian quién va ganando (un dragón aislado en el minuto 8, larvas).
- La caída del nexo al final: la partida ya estaba decidida antes. Marquen la jugada que la decidió.

## Reglas

1. **Entre 1 y 5 momentos por partida.** Si de verdad ninguno es claro (una partida pareja que se definió por desgaste), marquen la casilla *"No hay ningún momento claramente decisivo"*.
2. **El minuto es cuando ocurre la jugada**, no cuando se notan sus consecuencias. Si la pelea va de 23:10 a 24:05, el minuto es **23**.
3. **Una pelea larga es un solo momento.** Si un mismo enfrentamiento produce kills en los minutos 23 y 24, se marca una vez.
4. **Beneficia a:** el equipo que sale ganando de la jugada (aunque al final pierda la partida).
5. **Tipo:** `pelea`, `objetivo`, `pick`, `estructura`, `error` u `otro`. Si es pelea + objetivo (pelea que da el barón), elijan el que más pesó; en duda, `objetivo`.
6. **Confianza:** 3 = seguro, 2 = probable, 1 = dudoso. Los de confianza 1 se pueden excluir en un análisis de sensibilidad.
7. **Nota:** una frase de qué pasó ("ace en el río y barón", "pick a la ADC antes del elder"). Sirve para el informe y para revisar desacuerdos.
8. Pueden saber quién ganó (la herramienta lo muestra). Los momentos decisivos se definen respecto al resultado: es lo que buscaría un coach.
9. **Tiempo objetivo:** ~5–7 min por partida. La herramienta registra los segundos que tardaron (columna `segundos`).

## Tres ejemplos (patrones del log)

**1. Pelea + objetivo.** En el log:
```
24:12  🔴 rojo  Jinx (bottom) mató a Ahri (middle) (+3 asist.)      kills 12–14
24:20  🔴 rojo  Vi (jungle) mató a Ornn (top) (+2 asist.)           kills 12–15
24:31  🔴 rojo  Jinx (bottom): TRIPLE
24:58  🔴 rojo  BARÓN
```
→ Un momento: **minuto 24, `objetivo`, beneficia a rojo, confianza 3**, nota "pelea en el barón, triple de Jinx".

**2. Throw.** Azul va 15–6 en kills y 6–1 en torres en el minuto 28; en el 31 el log muestra 4 muertes de azul seguidas, un ace de rojo, un inhibidor de azul destruido y rojo gana en el 34.
→ **Minuto 31, `error`, beneficia a rojo, confianza 3**, nota "azul pierde pelea yendo adelante, rojo abre la base". (La caída del nexo en el 34 no se marca.)

**3. Partida decidida temprano.** Rendición en el minuto 15–16: en los minutos 5–9 hay dos peleas de 2–3 kills a favor de azul y azul se queda con los dos primeros dragones.
→ Uno o dos momentos tempranos (por ejemplo **minuto 6, `pelea`, azul**), aunque la partida haya durado poco. El EDA encontró que el 3.9 % de las partidas termina así.

## Qué se hace con las anotaciones

- **Partidas compartidas (15):** se mide el acuerdo entre A y B (% de momentos que coinciden a ±2 min). Si el acuerdo es bajo, el "techo" de cualquier detector también lo es, y eso se reporta.
- **Val (mitad de las partidas):** se elige el umbral θ del detector que maximiza el ahorro de tiempo.
- **Test (la otra mitad):** se reportan TP, FP, FN y el valor en dinero de cada modelo con ese θ.
- **Partidas propias:** si las agregan con `python src/select_annotation.py --add-own <match_id>...`, revisen además la repetición real en el cliente y **cronometren la revisión completa**: eso mide `f_rev`, la línea base del modelo de valor.
