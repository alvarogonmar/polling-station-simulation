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
        self.chosen_party = None
        self.vote_valid = None
        self.validation_started_at = None

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
        self.chosen_party = str(self.model.rng.choice(self.model.parties, p=probabilities))

    def cast_vote(self):
        self.vote_valid = self.model.rng.random() >= self.model.p_null_vote
        if self.vote_valid:
            self.choose_party()
        self.state = "finished"

    def should_abandon(self):
        return self.waiting_time > self.patience

    def to_dict(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "x": float(self.x),
            "y": float(self.y),
            "state": self.state,
            "age": int(self.age),
            "age_group": self.age_group,
            "waiting_time": int(self.waiting_time),
            "chosen_party": self.chosen_party,
            "vote_valid": self.vote_valid,
        }


class PollWorkerAgent(Agent):
    """Poll worker that validates and processes one voter at a time."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "poll_worker"
        self.state = "available"
        self.x = 6.0
        self.y = 0.0
        self.current_voter = None
        self.remaining_service_time = 0
        self.processed_voters = 0
        self.rejected_voters = 0

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
        }


class SupervisorAgent(Agent):
    """Supervisor that observes queue length and power failures."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "supervisor"
        self.state = "observing"
        self.x = 10.0
        self.y = 0.0
        self.interventions = 0

    def observe(self):
        if self.model.power_status == "outage":
            self.state = "intervening"
            return

        if len(self.model.queue) > self.model.queue_threshold:
            self.state = "intervening"
            self.interventions += 1
        else:
            self.state = "observing"

    def to_dict(self):
        return {
            "id": self.unique_id,
            "type": self.type,
            "x": float(self.x),
            "y": float(self.y),
            "state": self.state,
            "interventions": int(self.interventions),
        }
