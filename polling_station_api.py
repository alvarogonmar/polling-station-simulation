"""Flask API that exposes the Mesa polling station simulation."""

from flask import Flask, jsonify
from flasgger import Swagger

from polling_station_model import PollingStationModel


app = Flask(__name__)
swagger = Swagger(app)
model = PollingStationModel()


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
        }
    )


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
    model.step()
    return jsonify(model.to_json())


@app.route("/get_results", methods=["GET"])
def get_results():
    """
    Return accumulated simulation results without advancing the model.
    ---
    responses:
      200:
        description: Current accumulated statistics
    """
    return jsonify(model.get_results())


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
    model = PollingStationModel()
    return jsonify({"message": "Simulation reset", "step": model.step_count})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
