"""Mesa model for a polling station simulation."""

from collections import Counter
import heapq

import numpy as np
from mesa import Model
from mesa.time import RandomActivation

import config
from mesa_agents import BallotBoxAgent, PollWorkerAgent, SupervisorAgent, VoterAgent


ENTRANCE_POSITION = (2.0, 27.5)
EXIT_POSITION = (49.0, 25.0)
SUPERVISOR_POSITION = (8.5, 7.0)
VOTING_QUEUE_DELAY = 4.0
MAX_BALLOT_LANE_QUEUE_LENGTH = 8

GENERAL_QUEUE_POSITIONS = []
for row_index, y in enumerate([36.0, 33.5, 31.0, 28.5, 26.0, 23.5, 21.0, 18.5]):
    row_positions = [(5.5, y), (8.0, y), (10.5, y), (13.0, y)]
    if row_index % 2 == 1:
        row_positions.reverse()
    GENERAL_QUEUE_POSITIONS.extend(row_positions)

POLL_WORKER_POSITIONS = [
    (19.0, 34.0),
    (19.0, 27.0),
    (19.0, 20.0),
]

BALLOT_BOX_POSITIONS = [
    (46.0, 42.0),
    (46.0, 34.0),
    (46.0, 26.0),
    (46.0, 18.0),
    (46.0, 10.0),
]

VOTING_BOOTH_POSITIONS = [
    (39.8, 42.0),
    (39.8, 34.0),
    (39.8, 26.0),
    (39.8, 18.0),
    (39.8, 10.0),
]


class PollingStationModel(Model):
    """Central Mesa model that controls the polling station event queue."""

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(config.SEED)
        self.schedule = RandomActivation(self)
        self.step_count = 0
        self.simulation_clock = 0.0
        self.current_event = None
        self.event_queue = []
        self.event_log = []
        self.communication_log = []

        self.n_voters = config.N_VOTERS
        self.parties = config.PARTIES
        self.p_turnout = config.P_TURNOUT
        self.p_null_vote = config.P_NULL_VOTE
        self.p_id_invalid = config.P_ID_INVALID
        self.n_poll_workers = config.N_POLL_WORKERS
        self.n_ballot_boxes = config.N_BALLOT_BOXES
        self.queue_threshold = config.QUEUE_THRESHOLD
        self.mean_service_time = config.MEAN_SERVICE_TIME
        self.min_patience = config.MIN_PATIENCE
        self.max_patience = config.MAX_PATIENCE
        self.power_status = "normal"

        self.voter_line = []
        self.queue = self.voter_line
        self.ballot_box_queues = [[] for _ in range(self.n_ballot_boxes)]
        self.voters = self.create_voters()
        self.poll_workers = [
            PollWorkerAgent(501 + index, self)
            for index in range(self.n_poll_workers)
        ]
        for index, poll_worker in enumerate(self.poll_workers):
            poll_worker.x, poll_worker.y = POLL_WORKER_POSITIONS[index]
            self.schedule.add(poll_worker)
        self.ballot_boxes = [
            BallotBoxAgent(601 + index, self, index)
            for index in range(self.n_ballot_boxes)
        ]
        for index, ballot_box in enumerate(self.ballot_boxes):
            ballot_box.x, ballot_box.y = BALLOT_BOX_POSITIONS[index]
        for ballot_box in self.ballot_boxes:
            self.schedule.add(ballot_box)
        self.supervisor = SupervisorAgent(701, self)
        self.supervisor.x, self.supervisor.y = SUPERVISOR_POSITION
        self.schedule.add(self.supervisor)

        self.votes_by_party = Counter({party: 0 for party in self.parties})
        self.total_turnout = sum(voter.participates for voter in self.voters)
        self.total_abstention = self.n_voters - self.total_turnout
        self.total_valid_votes = 0
        self.total_null_votes = 0
        self.abandonment_count = 0
        self.max_queue_length = 0
        self.schedule_initial_events()

    def create_voters(self):
        voters = []
        for voter_id in range(1, self.n_voters + 1):
            age = int(self.rng.integers(18, 91))
            education = str(self.rng.choice(config.EDUCATION_LEVELS))
            income = str(self.rng.choice(config.INCOME_LEVELS))
            ideology = float(self.rng.uniform(-1.0, 1.0))

            voter = VoterAgent(voter_id, self, age, education, income, ideology)
            voter.decide_turnout()
            voters.append(voter)
            self.schedule.add(voter)
        return voters

    # --- issue-based affinity system (bloque activable/desactivable) ---
    def get_issue_affinity_scores(self, voter):
        if not config.ENABLE_ISSUE_AFFINITY or voter.issue_weights is None:
            return np.zeros(len(self.parties), dtype=float)

        voter_vector = np.array(voter.issue_weights, dtype=float)
        affinities = []
        for party in self.parties:
            platform = np.array(config.PARTY_PLATFORMS[party], dtype=float)
            l1_distance = np.abs(voter_vector - platform).sum()
            affinities.append(1.0 - (l1_distance / 200.0))
        return np.array(affinities, dtype=float)
    # --------------------------------------------------------------------

    def get_party_probabilities(self, voter):
        age_base = np.array(config.AGE_GROUP_PROBABILITIES[voter.age_group], dtype=float)
        education_effect = np.array(config.EDUCATION_EFFECTS[voter.education], dtype=float)
        income_effect = np.array(config.INCOME_EFFECTS[voter.income], dtype=float)
        ideology_effect = np.array(config.IDEOLOGY_EFFECTS, dtype=float) * voter.ideology

        # --- issue-based affinity system (bloque activable/desactivable) ---
        affinity_scores = self.get_issue_affinity_scores(voter)
        affinity_effect = (affinity_scores - 0.5) * config.AFFINITY_WEIGHT if config.ENABLE_ISSUE_AFFINITY else affinity_scores
        # --------------------------------------------------------------------

        utility_scores = (
            age_base + education_effect + income_effect + ideology_effect + affinity_effect
        )
        exp_scores = np.exp(utility_scores - np.max(utility_scores))
        probabilities = exp_scores / exp_scores.sum()
        return probabilities

    def schedule_initial_events(self):
        """Create initial ARRIVAL events before the simulation starts."""
        arrival_time = float(config.ARRIVAL_START_STEP)
        mean_interarrival_time = max(1.0, float(config.MEAN_INTERARRIVAL_TIME))

        for voter in self.voters:
            if not voter.participates:
                continue

            interarrival_time = self.rng.exponential(mean_interarrival_time)
            arrival_time += interarrival_time
            voter.arrival_time = arrival_time
            self.schedule_event(arrival_time, "ARRIVAL", voter.unique_id)

        self.schedule_event(config.POWER_FAILURE_STEP, "POWER_OUTAGE", None)
        self.schedule_event(
            config.POWER_FAILURE_STEP + config.POWER_FAILURE_DURATION,
            "POWER_RESTORED",
            None,
        )

    def schedule_event(self, event_time, event_type, voter_id):
        heapq.heappush(
            self.event_queue,
            (float(event_time), str(event_type), voter_id),
        )

    def step(self):
        if not self.event_queue:
            self.current_event = "FINISHED"
            return

        self.step_count += 1
        event_time, event_type, voter_id = heapq.heappop(self.event_queue)
        self.simulation_clock = event_time
        self.current_event = event_type

        self.process_event(event_type, voter_id)
        self.supervisor.observe()
        self.update_queue_positions()
        self.max_queue_length = max(self.max_queue_length, len(self.queue))
        self.event_log.append(
            {
                "time": round(float(event_time), 2),
                "event": event_type,
                "voter_id": voter_id,
            }
        )

    def process_event(self, event_type, voter_id):
        if event_type == "POWER_OUTAGE":
            self.power_status = "outage"
            for poll_worker in self.poll_workers:
                poll_worker.state = "paused"
            self.communication_log.append(
                "SupervisorAgent -> PollWorkerAgents: pause service because of POWER_OUTAGE"
            )
            return

        if event_type == "POWER_RESTORED":
            self.power_status = "normal"
            for poll_worker in self.poll_workers:
                if poll_worker.current_voter is None:
                    poll_worker.state = "available"
                else:
                    poll_worker.state = "busy"
            self.communication_log.append(
                "SupervisorAgent -> PollWorkerAgents: resume service after POWER_RESTORED"
            )
            self.try_start_next_validation()
            self.try_start_next_voting()
            return

        voter = self.get_voter_by_id(voter_id)
        if voter is None or voter.state in {"abandoned", "finished", "rejected"}:
            return
        if self.power_status == "outage":
            self.schedule_event(self.simulation_clock + 1.0, event_type, voter_id)
            return

        if event_type == "ARRIVAL":
            self.process_arrival(voter)
        elif event_type == "VALIDATION":
            self.process_validation(voter)
        elif event_type == "VOTING":
            self.process_voting(voter)
        elif event_type == "START_VOTING":
            self.process_start_voting()
        elif event_type == "EXIT":
            self.process_exit(voter)

    def process_arrival(self, voter):
        voter.state = "waiting"
        voter.x, voter.y = ENTRANCE_POSITION
        self.queue.append(voter)
        self.communication_log.append(
            f"VoterAgent {voter.unique_id} -> PollWorkerAgents: joins line and waits for validation"
        )
        self.try_start_next_validation()

    def process_validation(self, voter):
        poll_worker = self.get_poll_worker_for_voter(voter)
        if poll_worker is None or poll_worker.current_voter != voter:
            return

        voter.waiting_time = int(round(self.simulation_clock - voter.arrival_time))
        if voter.should_abandon():
            poll_worker.release_voter()
            voter.assigned_poll_worker = None
            voter.state = "abandoned"
            voter.x, voter.y = EXIT_POSITION
            self.abandonment_count += 1
            self.try_start_next_validation()
            return

        voter.state = "validating"
        voter.x = poll_worker.x - 2.4
        voter.y = poll_worker.y

        id_valid = self.rng.random() >= self.p_id_invalid
        voter.receive_validation_result(id_valid)
        self.communication_log.append(
            f"PollWorkerAgent {poll_worker.unique_id} -> VoterAgent {voter.unique_id}: validation result = {id_valid}"
        )
        if not id_valid:
            voter.state = "rejected"
            poll_worker.rejected_voters += 1
            poll_worker.processed_voters += 1
            poll_worker.release_voter()
            voter.assigned_poll_worker = None
            self.try_start_next_validation()
            return

        voter.state = "ready_to_vote"
        ballot_box = self.get_ballot_box_with_shortest_line()
        if ballot_box is None:
            voter.state = "validating"
            self.schedule_event(
                self.simulation_clock + 1.0,
                "VALIDATION",
                voter.unique_id,
            )
            return

        voter.assigned_ballot_box = ballot_box.unique_id
        voter.ready_to_vote_at = self.simulation_clock
        self.ballot_box_queues[self.get_ballot_box_index(ballot_box)].append(voter)
        self.update_ballot_box_queue_positions()
        poll_worker.processed_voters += 1
        poll_worker.release_voter()
        voter.assigned_poll_worker = None
        self.try_start_next_validation()
        self.schedule_event(
            self.simulation_clock + VOTING_QUEUE_DELAY,
            "START_VOTING",
            voter.unique_id,
        )
        self.try_start_next_voting()

    def process_start_voting(self):
        self.try_start_next_voting()

    def process_voting(self, voter):
        ballot_box = self.get_ballot_box_for_voter(voter)
        if ballot_box is None or ballot_box.current_voter != voter:
            return

        voter.state = "voting"
        ballot_box_index = self.get_ballot_box_index(ballot_box)
        voter.x, voter.y = VOTING_BOOTH_POSITIONS[ballot_box_index]
        voter.cast_vote()
        self.register_vote(voter)
        ballot_box.processed_voters += 1
        ballot_box.release_voter()
        voter.assigned_ballot_box = None
        self.schedule_event(self.simulation_clock + 1.0, "EXIT", voter.unique_id)
        self.try_start_next_voting()

    def process_exit(self, voter):
        voter.state = "finished"
        voter.x, voter.y = EXIT_POSITION

    def update_power_status(self):
        outage_start = config.POWER_FAILURE_STEP
        outage_end = outage_start + config.POWER_FAILURE_DURATION
        if outage_start <= self.step_count < outage_end:
            self.power_status = "outage"
        else:
            self.power_status = "normal"

    def add_arrivals_to_queue(self):
        for voter in self.voters:
            if voter.state == "outside" and voter.participates:
                if voter.arrival_time <= self.step_count:
                    voter.state = "waiting"
                    self.queue.append(voter)

    def update_waiting_voters(self):
        remaining_queue = []
        for voter in self.queue:
            voter.waiting_time += 1
            if voter.should_abandon():
                voter.state = "abandoned"
                voter.x, voter.y = EXIT_POSITION
                self.abandonment_count += 1
            else:
                remaining_queue.append(voter)
        self.queue = remaining_queue

    def update_queue_positions(self):
        for index, voter in enumerate(self.queue):
            voter.x, voter.y = self.get_general_queue_position(index)
        self.update_ballot_box_queue_positions()

    def update_ballot_box_queue_positions(self):
        for ballot_box_index, ballot_box_queue in enumerate(self.ballot_box_queues):
            ballot_box = self.ballot_boxes[ballot_box_index]
            for queue_index, voter in enumerate(ballot_box_queue):
                voter.x = ballot_box.x - 7.5 - float(queue_index * 2.0)
                voter.y = ballot_box.y

    def get_general_queue_position(self, index):
        if index < len(GENERAL_QUEUE_POSITIONS):
            return GENERAL_QUEUE_POSITIONS[index]

        overflow_index = index - len(GENERAL_QUEUE_POSITIONS)
        return (
            ENTRANCE_POSITION[0],
            max(2.0, ENTRANCE_POSITION[1] - float(overflow_index * 1.3)),
        )

    def try_start_next_validation(self):
        if self.power_status == "outage":
            for poll_worker in self.poll_workers:
                poll_worker.state = "paused"
            return

        for poll_worker in self.poll_workers:
            if poll_worker.current_voter is not None:
                continue
            if not self.has_ballot_lane_capacity():
                poll_worker.state = "available"
                continue
            if not self.queue:
                poll_worker.state = "available"
                continue

            voter = self.queue.pop(0)
            voter.state = "validating"
            voter.x = poll_worker.x - 2.4
            voter.y = poll_worker.y
            voter.assigned_poll_worker = poll_worker.unique_id
            voter.request_validation(poll_worker)

            service_delay = self.rng.exponential(self.mean_service_time)
            validation_time = self.simulation_clock + max(1.0, service_delay)
            self.schedule_event(validation_time, "VALIDATION", voter.unique_id)
            self.communication_log.append(
                f"PollWorkerAgent {poll_worker.unique_id} -> VoterAgent {voter.unique_id}: validation scheduled"
            )

    def try_start_next_voting(self):
        if self.power_status == "outage":
            return

        for ballot_box in self.ballot_boxes:
            if ballot_box.current_voter is not None:
                continue
            ballot_box_queue = self.ballot_box_queues[self.get_ballot_box_index(ballot_box)]
            if not ballot_box_queue:
                ballot_box.state = "available"
                continue

            first_voter = ballot_box_queue[0]
            if first_voter.ready_to_vote_at is not None:
                wait_time = self.simulation_clock - first_voter.ready_to_vote_at
                if wait_time < VOTING_QUEUE_DELAY:
                    ballot_box.state = "available"
                    continue

            voter = ballot_box_queue.pop(0)
            voter.state = "voting"
            ballot_box_index = self.get_ballot_box_index(ballot_box)
            voter.x, voter.y = VOTING_BOOTH_POSITIONS[ballot_box_index]
            ballot_box.receive_voter(voter)

            voting_duration = self.rng.uniform(config.MIN_VOTING_TIME, config.MAX_VOTING_TIME)
            voting_time = self.simulation_clock + voting_duration
            self.schedule_event(voting_time, "VOTING", voter.unique_id)
            self.communication_log.append(
                f"BallotBoxAgent {ballot_box.unique_id} -> VoterAgent {voter.unique_id}: voting scheduled"
            )

        self.update_ballot_box_queue_positions()
        self.try_start_next_validation()

    def get_ballot_box_with_shortest_line(self):
        ballot_boxes_with_capacity = [
            ballot_box
            for ballot_box in self.ballot_boxes
            if len(self.ballot_box_queues[self.get_ballot_box_index(ballot_box)])
            < MAX_BALLOT_LANE_QUEUE_LENGTH
        ]

        if not ballot_boxes_with_capacity:
            return None

        return min(
            ballot_boxes_with_capacity,
            key=lambda ballot_box: len(self.ballot_box_queues[self.get_ballot_box_index(ballot_box)])
            + (1 if ballot_box.current_voter is not None else 0),
        )

    def has_ballot_lane_capacity(self):
        return any(
            len(ballot_box_queue) < MAX_BALLOT_LANE_QUEUE_LENGTH
            for ballot_box_queue in self.ballot_box_queues
        )

    def get_ballot_box_index(self, ballot_box):
        return int(ballot_box.unique_id - 601)

    def get_poll_worker_for_voter(self, voter):
        for poll_worker in self.poll_workers:
            if poll_worker.unique_id == voter.assigned_poll_worker:
                return poll_worker
        return None

    def get_ballot_box_for_voter(self, voter):
        for ballot_box in self.ballot_boxes:
            if ballot_box.unique_id == voter.assigned_ballot_box:
                return ballot_box
        return None

    def get_voter_by_id(self, voter_id):
        if voter_id is None:
            return None
        return self.voters[int(voter_id) - 1]

    def register_vote(self, voter):
        if voter.vote_valid:
            self.total_valid_votes += 1
            self.votes_by_party[voter.chosen_party] += 1
        else:
            self.total_null_votes += 1

    def get_stats(self):
        finished_voters = sum(1 for voter in self.voters if voter.state == "finished")
        rejected_voters = sum(1 for voter in self.voters if voter.state == "rejected")
        active_voters = sum(
            1
            for voter in self.voters
            if voter.state in {"waiting", "validating", "ready_to_vote", "voting"}
        )
        pending_arrivals = sum(
            1
            for voter in self.voters
            if voter.participates and voter.state == "outside"
        )

        return {
            "total_registered": self.n_voters,
            "total_turnout": int(self.total_turnout),
            "total_abstention": int(self.total_abstention),
            "valid_votes": int(self.total_valid_votes),
            "null_votes": int(self.total_null_votes),
            "finished_voters": int(finished_voters),
            "rejected_voters": int(rejected_voters),
            "active_voters": int(active_voters),
            "pending_arrivals": int(pending_arrivals),
            "queue_length": len(self.queue),
            "ready_to_vote_queue_length": sum(len(queue) for queue in self.ballot_box_queues),
            "ballot_box_queue_lengths": [
                len(queue) for queue in self.ballot_box_queues
            ],
            "max_queue_length": int(self.max_queue_length),
            "abandonment_count": int(self.abandonment_count),
            "poll_workers": self.n_poll_workers,
            "ballot_boxes": self.n_ballot_boxes,
            "power_status": self.power_status,
            "event_queue_size": len(self.event_queue),
            "main_events": ["ARRIVAL", "VALIDATION", "VOTING", "EXIT"],
            "external_event": "POWER_OUTAGE",
            "votes_by_party": dict(self.votes_by_party),
        }

    def get_agent_payload(self):
        active_voters = [
            voter
            for voter in self.voters
            if voter.participates
            and voter.state in {"waiting", "validating", "ready_to_vote", "voting"}
        ]
        visible_voters = active_voters[:80]
        return (
            [voter.to_dict() for voter in visible_voters]
            + [poll_worker.to_dict() for poll_worker in self.poll_workers]
            + [ballot_box.to_dict() for ballot_box in self.ballot_boxes]
            + [self.supervisor.to_dict()]
        )

    def to_json(self):
        return {
            "step": self.step_count,
            "simulation_clock": round(float(self.simulation_clock), 2),
            "current_event": self.current_event,
            "current_event_queue_size": len(self.event_queue),
            "main_events": ["ARRIVAL", "VALIDATION", "VOTING", "EXIT"],
            "external_event": "POWER_OUTAGE",
            "communication_log": self.communication_log[-8:],
            "agents": self.get_agent_payload(),
            "stats": self.get_stats(),
        }

    def get_results(self):
        return self.get_stats()


if __name__ == "__main__":
    model = PollingStationModel()
    for _ in range(10):
        model.step()
    print(model.to_json())
