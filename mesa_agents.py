"""Mesa agents for the polling station simulation."""

import numpy as np
from mesa import Agent


class VoterAgent(Agent):
    """Voter with demographic attributes and voting decisions."""

    def __init__(self, unique_id, model, age, education, income, ideology):
        super().__init__(unique_id, model)
        self.type = "voter"
        self.age = age
        self.age_group = self._get_age_group(age)
        self.education = education
        self.income = income
        self.ideology = ideology
        self.state = "outside"
        self.x = 0.0
        self.y = 0.0
        self.arrival_time = None
        self.waiting_time = 0
        self.patience = model.rng.integers(model.min_patience, model.max_patience + 1)
        self.participates = False
        self.party_probabilities = None
        self.chosen_party = None
        self.vote_valid = None
        self.validation_started_at = None
        self.ready_to_vote_at = None
        self.assigned_poll_worker = None
        self.assigned_ballot_box = None
        self.last_message = None

    def _get_age_group(self, age):
        if age <= 29:
            return "18-29"
        if age <= 44:
            return "30-44"
        if age <= 59:
            return "45-59"
        return "60+"

    def decide_turnout(self):
        self.participates = self.model.rng.random() < self.model.p_turnout
        if not self.participates:
            self.state = "abstention"

    def choose_party(self):
        probabilities = self.model.get_party_probabilities(self)
        self.party_probabilities = [round(float(probability), 4) for probability in probabilities]
        self.chosen_party = str(self.model.rng.choice(self.model.parties, p=probabilities))

    def cast_vote(self):
        self.vote_valid = self.model.rng.random() >= self.model.p_null_vote
        if self.vote_valid:
            self.choose_party()

    def should_abandon(self):
        return self.waiting_time > self.patience

    def request_validation(self, poll_worker):
        self.last_message = f"Voter {self.unique_id} requests validation"
        poll_worker.receive_voter(self)

    def receive_validation_result(self, is_valid):
        if is_valid:
            self.last_message = "Poll worker approved voter ID"
        else:
            self.last_message = "Poll worker rejected voter ID"

    def to_dict(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "x": float(self.x),
            "y": float(self.y),
            "state": self.state,
            "age": int(self.age),
            "age_group": self.age_group,
            "education": self.education,
            "income": self.income,
            "ideology": round(float(self.ideology), 4),
            "party_probabilities": self.party_probabilities,
            "waiting_time": int(self.waiting_time),
            "chosen_party": self.chosen_party,
            "vote_valid": self.vote_valid,
            "assigned_poll_worker": self.assigned_poll_worker,
            "assigned_ballot_box": self.assigned_ballot_box,
            "last_message": self.last_message,
        }


class PollWorkerAgent(Agent):
    """Poll worker that validates and processes one voter at a time."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "poll_worker"
        self.state = "available"
        self.x = 19.0
        self.y = 27.0
        self.current_voter = None
        self.remaining_service_time = 0
        self.processed_voters = 0
        self.rejected_voters = 0
        self.last_message = None

    def receive_voter(self, voter):
        self.current_voter = voter
        self.state = "busy"
        self.last_message = f"Received validation request from voter {voter.unique_id}"

    def release_voter(self):
        self.current_voter = None
        self.state = "available"

    def process(self):
        if self.model.power_status == "outage":
            self.state = "paused"
            return

        if self.current_voter is not None:
            self.remaining_service_time -= 1
            if self.remaining_service_time <= 0:
                self.finish_current_voter()
            return

        if self.model.queue:
            self.current_voter = self.model.queue.pop(0)
            self.current_voter.state = "validating"
            self.current_voter.x = self.x - 1.0
            self.current_voter.y = self.y
            self.remaining_service_time = max(
                1,
                int(round(self.model.rng.exponential(self.model.mean_service_time))),
            )
            self.state = "busy"
        else:
            self.state = "available"

    def finish_current_voter(self):
        voter = self.current_voter
        self.current_voter = None
        self.processed_voters += 1

        id_valid = self.model.rng.random() >= self.model.p_id_invalid
        if not id_valid:
            voter.state = "rejected"
            self.rejected_voters += 1
        else:
            voter.state = "voting"
            voter.x = 8.0
            voter.y = 0.0
            voter.cast_vote()
            self.model.register_vote(voter)

        self.state = "available"

    def to_dict(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "x": float(self.x),
            "y": float(self.y),
            "state": self.state,
            "processed_voters": int(self.processed_voters),
            "rejected_voters": int(self.rejected_voters),
            "last_message": self.last_message,
        }


class SupervisorAgent(Agent):
    """Supervisor that observes queue length and power failures."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "supervisor"
        self.state = "observing"
        self.x = 8.5
        self.y = 7.0
        self.interventions = 0
        self.last_message = None

    def observe(self):
        if self.model.power_status == "outage":
            self.state = "intervening"
            self.last_message = "External event detected: power outage"
            return

        if len(self.model.queue) > self.model.queue_threshold:
            self.state = "intervening"
            self.interventions += 1
            self.last_message = "Queue is too long; poll worker should prioritize service"
        else:
            self.state = "observing"
            self.last_message = "Polling station operating normally"

    def to_dict(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "x": float(self.x),
            "y": float(self.y),
            "state": self.state,
            "interventions": int(self.interventions),
            "last_message": self.last_message,
        }


class BallotBoxAgent(Agent):
    """Voting booth or ballot box that can be used by one voter at a time."""

    def __init__(self, unique_id, model, lane_index):
        super().__init__(unique_id, model)
        self.type = "ballot_box"
        self.state = "available"
        self.x = 46.0
        self.y = 26.0
        self.current_voter = None
        self.processed_voters = 0
        self.last_message = None

    def receive_voter(self, voter):
        self.current_voter = voter
        self.state = "busy"
        self.last_message = f"Voter {voter.unique_id} is voting here"

    def release_voter(self):
        self.current_voter = None
        self.state = "available"

    def to_dict(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "x": float(self.x),
            "y": float(self.y),
            "state": self.state,
            "processed_voters": int(self.processed_voters),
            "last_message": self.last_message,
        }
