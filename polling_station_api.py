"""Flask API and live dashboard for the polling station simulation."""

from threading import Lock # Lock sirve para evitar que varios hilos accedan a la misma variable al mismo tiempo, lo que podría causar errores o resultados inesperados.

from flask import Flask, jsonify, render_template, request
from flasgger import Swagger

from polling_station_model import PollingStationModel


app = Flask(__name__)
swagger = Swagger(app)
model = PollingStationModel()
model_lock = Lock()


@app.route("/", methods=["GET"])
def home():
    """
    Check that the polling station API is running.
    ---
    responses:
      200:
        description: API status message
    """
    return jsonify(
        {
            "message": "Polling station API is running",
            "docs": "http://127.0.0.1:5000/apidocs",
            "main_endpoint": "http://127.0.0.1:5000/get_agents",
            "dashboard": "http://127.0.0.1:5000/dashboard",
        }
    )


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Render the live, read-only results dashboard."""
    return render_template("dashboard.html")


@app.route("/get_agents", methods=["GET"])
def get_agents():
    """
    Process the next chronological simulation event and return agent states.
    ---
    responses:
      200:
        description: Current simulation event, visible agents, and statistics
        schema:
          type: object
          properties:
            step:
              type: integer
            simulation_clock:
              type: number
            current_event:
              type: string
            current_event_queue_size:
              type: integer
            agents:
              type: array
              items:
                type: object
            stats:
              type: object
    """
    with model_lock:
        model.step()
        payload = model.to_json()
    return jsonify(payload)


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """Return dashboard metrics without advancing the simulation."""
    requested_points = request.args.get("max_points", default=600, type=int)
    max_points = min(max(requested_points, 50), 2000)
    with model_lock:
        payload = model.get_dashboard_data(max_points)
    return jsonify(payload)


@app.route("/get_results", methods=["GET"])
def get_results():
    """
    Return accumulated simulation results without advancing the model.
    ---
    responses:
      200:
        description: Current accumulated statistics
    """
    with model_lock:
        payload = model.get_results()
    return jsonify(payload)


@app.route("/reset", methods=["POST"])
def reset():
    """
    Reset the Mesa simulation.
    ---
    responses:
      200:
        description: Reset confirmation
    """
    global model
    with model_lock:
        model = PollingStationModel()
    return jsonify({"message": "Simulation reset", "step": model.step_count})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
