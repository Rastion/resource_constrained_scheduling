from qubots.base_problem import BaseProblem
import random, os

class RCPSPProblem(BaseProblem):
    """
    Resource Constrained Project Scheduling Problem (RCPSP).

    Instance Format (Patterson format):
      - First line: two integers: number of tasks (nb_tasks) and number of renewable resources (nb_resources).
      - Second line: maximum capacity for each resource (list of integers).
      - Next nb_tasks lines: for each task:
           • duration (integer),
           • resource requirements for each resource (nb_resources integers),
           • number of successors (integer),
           • successor task IDs (1-indexed).
    
    A candidate solution is represented as a list of start times (integers) for each task.
    The objective is to minimize the makespan (the maximum finish time among all tasks).

    Constraints:
      - Precedence: For every task i and each successor j, the start time of j must be at least start[i] + duration[i].
      - Resource: For every time unit t, the total resource usage (from tasks active at t) must not exceed the resource capacity.
    
    Infeasible solutions are penalized heavily.
    """
    
    def __init__(self, instance_file):
        self.instance_file = instance_file
        self._load_instance(instance_file)
        self.penalty_multiplier = 1000000  # A large number to penalize constraint violations

    def _load_instance(self, filename):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            lines = f.readlines()
        
        # First line: number of tasks and number of resources
        first_line = lines[0].split()
        self.nb_tasks = int(first_line[0])
        self.nb_resources = int(first_line[1])
        
        # Second line: resource capacities
        self.capacity = [int(x) for x in lines[1].split()[:self.nb_resources]]
        
        # Initialize lists for task data
        self.duration = [0] * self.nb_tasks
        self.weight = [[] for _ in range(self.nb_tasks)]
        self.nb_successors = [0] * self.nb_tasks
        self.successors = [[] for _ in range(self.nb_tasks)]
        
        # Each subsequent line describes one task
        for i in range(self.nb_tasks):
            tokens = lines[i + 2].split()
            self.duration[i] = int(tokens[0])
            self.weight[i] = [int(tokens[r + 1]) for r in range(self.nb_resources)]
            self.nb_successors[i] = int(tokens[self.nb_resources + 1])
            # Convert successor task IDs from 1-indexed to 0-indexed.
            self.successors[i] = [int(tokens[self.nb_resources + 2 + s]) - 1 for s in range(self.nb_successors[i])]
        
        # Compute a trivial upper bound for the makespan as the sum of durations.
        self.horizon = sum(self.duration)

    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Parameters:
            solution: A list of start times (one integer per task).
        
        Returns:
            A cost equal to the makespan if all constraints are met.
            If any constraint is violated, a penalty is added.
        """
        if len(solution) != self.nb_tasks:
            raise ValueError("Solution must have a start time for each task.")
        
        penalty = 0
        
        # Check precedence constraints.
        for i in range(self.nb_tasks):
            finish_i = solution[i] + self.duration[i]
            for succ in self.successors[i]:
                if solution[succ] < finish_i:
                    penalty += (finish_i - solution[succ])
        
        # Determine the makespan as the maximum finish time.
        makespan = max(solution[i] + self.duration[i] for i in range(self.nb_tasks))
        
        # Check resource constraints at each time unit.
        for t in range(makespan):
            for r in range(self.nb_resources):
                usage = 0
                # A task i is active at time t if its start time <= t < start time + duration.
                for i in range(self.nb_tasks):
                    if solution[i] <= t < solution[i] + self.duration[i]:
                        usage += self.weight[i][r]
                if usage > self.capacity[r]:
                    penalty += (usage - self.capacity[r])
        
        # The total cost is the makespan plus a heavy penalty for any constraint violations.
        cost = makespan + self.penalty_multiplier * penalty
        return cost

    def random_solution(self):
        """
        Generates a random candidate solution by assigning each task
        a random start time between 0 and the horizon.
        """
        return [random.randint(0, self.horizon) for _ in range(self.nb_tasks)]
