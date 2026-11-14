# Despliegue de Fight Diff en Render (B13)

*13 nov 2026. Configuración en `render.yaml`; dependencias en `requirements-deploy.txt`.*

## Por qué Render (plan gratis) y qué esperar

| | Valor |
|---|---|
| RAM del plan | 512 MB |
| RAM de la app | ~240 MB al arrancar, ~260 MB de pico (medido con PyTorch CPU tras 10 análisis). Los modelos del selector se cargan al primer uso: M0, M1, M3 y M5 suman < 1 MB; M2 (LightGBM + scipy) ~27 MB |
| CPU del plan | 0.1 CPU: el análisis tarda más que en una laptop (la API de Riot sigue dominando) |
| Inactividad | se "duerme" tras 15 min sin visitas; la primera visita después tarda ~1 min en despertar |
| Disco | efímero: la caché y `predicciones.jsonl` se pierden en cada reinicio o redeploy |
| Costo | USD 0/mes (es el `C_infra` del modelo de valor) |

Con PyTorch CUDA la app usaba ~490 MB y no cabía; por eso `requirements-deploy.txt` instala la versión **CPU**.

---

## Antes de empezar (una sola vez)

1. **Confirmar con Melisa qué API key es.**
   - **Personal API key** (del producto registrado): no vence. ✅ Es la que debe usar la app pública.
   - **Development key**: vence cada 24 h. Sirve para probar, pero el día de la demo hay que regenerarla y actualizarla (paso 7).
2. **Tener a mano los valores de `RIOT_API_KEY` y `PUUID_SALT`** del `.env` que te pasó Melisa. No se escriben en ningún archivo del repo: se pegan en el panel de Render.
3. **Subir el código a GitHub** (Render lo toma del repo público). Desde la raíz del repo:
   ```powershell
   git add models .gitignore requirements-deploy.txt render.yaml docs/despliegue.md
   git commit -m "Despliegue: modelo de la app, dependencias CPU y configuracion de Render"
   git pull
   git push
   ```
   Verifica en https://github.com/melisadmendizabal/deep_proyecto4 que aparezcan `render.yaml` y las carpetas de `models/`: `m6_draft` (el principal: Transformer + campeones) y `m4_draft`, `m4_gru`, `m0_oro`, `m1_logit`, `m2_lgbm`, `m3_mlp`, `m5_tcn` (selector de modelos). `models/_train/` NO debe subirse.

---

## Pasos en Render

### 1. Crear el servicio
1. Entra a https://dashboard.render.com e inicia sesión.
2. **New +** → **Web Service**.
3. Elige **Public Git Repository** y pega `https://github.com/melisadmendizabal/deep_proyecto4`. Clic en **Connect**.
   *El repo es público, así que no hace falta que Melisa te dé permisos ni conectar GitHub.*

### 2. Configuración básica
| Campo | Valor |
|---|---|
| Name | `fight-diff` (la URL quedará `https://fight-diff.onrender.com`, o con un sufijo si el nombre está ocupado) |
| Region | cualquiera de EE. UU. (Oregon u Ohio) |
| Branch | `main` |
| Root Directory | *(vacío)* |
| Runtime / Language | `Python 3` |
| Build Command | `pip install -r requirements-deploy.txt` |
| Start Command | `uvicorn app.api:app --host 0.0.0.0 --port $PORT --workers 1` |
| Instance Type | **Free** |

### 3. Variables de entorno
En **Environment Variables** → **Add Environment Variable**, una por una:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.12.10` |
| `OMP_NUM_THREADS` | `1` |
| `MKL_NUM_THREADS` | `1` |
| `RIOT_API_KEY` | *(la key del `.env`, empieza con `RGAPI-`)* |
| `PUUID_SALT` | *(el valor del `.env`)* |

### 4. Opciones avanzadas
En **Advanced**:
- **Health Check Path**: `/health`
- **Auto-Deploy**: con un repo público agregado por URL, Render no recibe avisos de cada push. Cada actualización se despliega a mano (paso 8).

### 5. Crear y esperar el build
Clic en **Create Web Service**. En **Logs** deberías ver, en orden:
1. `Installing Python version 3.12.10`
2. la instalación de dependencias (la de `torch` es la más pesada; el build tarda ~3–6 min);
3. `Build successful 🎉`;
4. `Uvicorn running on http://0.0.0.0:10000` y luego `Your service is live 🎉`.

### 6. Verificar
1. Abre `https://<tu-servicio>.onrender.com/health`. Debe responder un JSON con:
   - `"ok": true`
   - `"modelo": "M6"`, `"theta": 0.15`
   - `"modelos"` con 8 entradas (M6 primero, `"principal": true`).
   - `"api_riot": true` (si sale `false`, revisa `RIOT_API_KEY`)
   - `"partidas_demo"` con 5 partidas.
2. Abre `https://<tu-servicio>.onrender.com/`:
   - Busca `BadLuckLegend#NA1` (servidor NA): deben salir tus 5 partidas con el ícono del campeón.
   - Haz clic en una: la curva y los momentos deben aparecer en unos segundos.
   - Cambia el **Modelo** a M2 (LightGBM): la primera vez tarda un poco más (carga el modelo).
   - Prueba la pestaña **Uso sin conexión**: elige una partida de ejemplo y dale Analizar.
3. Prueba un error: Match ID `hola` → debe mostrar el mensaje de ID inválido.
4. En **Metrics** de Render revisa la memoria: debería rondar 250–300 MB.

### 7. Cambiar la API key (si es development key, cada día de uso)
1. Genera una nueva en https://developer.riotgames.com.
2. Render → tu servicio → **Environment** → edita `RIOT_API_KEY` → **Save Changes**. El servicio se reinicia solo (~1 min).

### 8. Actualizar la app después de un cambio de código o de modelo
Si cambian θ* o reentrenan un modelo (comandos completos en el encabezado de `src/export_model.py`):
```powershell
# el principal (M6 + campeones, promedio de 3 semillas):
python src/export_model.py --runs results/runs/m6d_final__s42 results/runs/m6d_final__s43 results/runs/m6d_final__s44 --name "M6+draft ens3" --manifest-name M6 --out models/m6_draft --promedio --theta 0.15
python app/make_demo.py             # rehace las partidas de ejemplo con el modelo nuevo
# los demás modelos del selector: ver los comandos en el encabezado de src/export_model.py
git add models && git commit -m "Modelo de la app actualizado" && git push
```
Luego, en Render: **Manual Deploy** → **Deploy latest commit**.

---

## Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| El build falla en `torch==2.6.0+cpu` | falta la línea `--extra-index-url` o la versión de Python no es 3.12 | revisar `requirements-deploy.txt` y `PYTHON_VERSION` |
| `/health` dice `"api_riot": false` | `RIOT_API_KEY` vacía o mal pegada | corregir la variable (paso 7) |
| Buscar un Riot ID da 503 "la API key es inválida o expiró" | development key vencida | regenerar y actualizar (paso 7) |
| Logs con `Out of memory` o reinicios | se pasó de 512 MB | revisar **Metrics**; plan B: Hugging Face Spaces (16 GB gratis) |
| La primera visita tarda ~1 min | el servicio estaba dormido | normal en el plan gratis: abrir la app unos minutos antes de la demo |
| `No existe models/m6_draft/manifest.json` | no se subieron los pesos | revisar el paso 3 de "Antes de empezar" |
| El selector solo muestra M4, o M2 da error 500 | no se subieron `models/m*/`, o falta `lightgbm` en el build | revisar el paso 3 y `requirements-deploy.txt` |

## El día de la demo
1. **10 min antes:** abrir la URL para despertar el servicio y analizar una partida.
2. **Plan B (sin internet o con Render caído):** correr local en la laptop:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   uvicorn app.api:app --port 8000
   ```
   y abrir http://127.0.0.1:8000.
3. **Plan C (sin API de Riot):** las 5 "Partidas de ejemplo" funcionan siempre, porque están precalculadas en `app/demo/`.
