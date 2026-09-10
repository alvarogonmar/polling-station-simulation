# Polling Station Simulation

Simulacion multiagente de una casilla electoral usando Mesa como motor, Flask como API, C# como cliente de prueba y Unity como visualizacion.

## Setup

Crear y activar entorno virtual :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Estructura Planeada

```text
polling_station_api.py      # Flask + Swagger
polling_station_model.py    # Modelo Mesa
mesa_agents.py              # VoterAgent, PollWorkerAgent, SupervisorAgent
config.py                   # Parametros del modelo
statistics.py               # Resultados, CSV y graficas
csharp_client/              # Cliente C# para probar Flask
unity/                      # Scripts o notas de integracion con Unity
outputs/                    # Archivos generados localmente
```

## Ejecucion Esperada

Cuando exista `polling_station_api.py`:

```bash
source .venv/bin/activate
python polling_station_api.py
```

Luego abrir:

```text
http://127.0.0.1:5000/apidocs
```

Dashboard de resultados en tiempo real:

```text
http://127.0.0.1:5000/dashboard
```

El dashboard consulta `GET /dashboard_data`, que es de solo lectura. Unity sigue
usando `GET /get_agents` para avanzar la simulacion, por lo que abrir el dashboard
no modifica la velocidad ni el orden de los eventos.

Endpoint principal:

```text
GET http://127.0.0.1:5000/get_agents
```
