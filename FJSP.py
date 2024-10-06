import numpy as np
from mealpy.bio_based import BatAlgorithm as BA
from mealpy.encoding import PermutationVar
from mealpy.problem import Problem

# Function to read job and machine times from a file
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Read number of jobs and machines
        n_jobs, n_machines = map(int, lines[0].strip().split())
        
        job_times = np.zeros((n_jobs, n_machines))
        
        current_line = 1
        for job_idx in range(n_jobs):
            operation_info = list(map(int, lines[current_line].strip().split()))
            num_operations = operation_info[0]
            operations = operation_info[1:]
            
            # Process each operation for the current job
            for op_idx in range(num_operations):
                machine_idx = operations[2 * op_idx]
                processing_time = operations[2 * op_idx + 1]
                job_times[job_idx][machine_idx] = processing_time
            
            current_line += 1
        
        return job_times, n_jobs, n_machines

# Read data from Input.txt
job_times, n_jobs, n_machines = read_data_from_file("Input.txt")

data = {
    "job_times": job_times,
    "n_jobs": n_jobs,
    "n_machines": n_machines
}

class FlexibleJobShopProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["per_var"]
        
        # Initialize makespan matrix
        job_end_times = np.zeros(self.data["n_jobs"])
        machine_end_times = np.zeros(self.data["n_machines"])

        for gene in x:
            job_idx = gene // self.data["n_machines"]
            machine_idx = gene % self.data["n_machines"]
            
            start_time = max(job_end_times[job_idx], machine_end_times[machine_idx])
            finish_time = start_time + self.data["job_times"][job_idx][machine_idx]
            
            job_end_times[job_idx] = finish_time
            machine_end_times[machine_idx] = finish_time
            
        return np.max(job_end_times)

bounds = PermutationVar(valid_set=list(range(0, n_jobs*n_machines)), name="per_var")
problem = FlexibleJobShopProblem(bounds=bounds, minmax="min", data=data)

# Initialize and solve using Bat Algorithm (BA)
ba_model = BA.OriginalBA(epoch=50, pop_size=10)
ba_model.solve(problem)

# Get the best solution from BA
best_ba_solution = ba_model.g_best.solution

print(f"Best agent from BA: {ba_model.g_best}")
print(f"Best solution from BA: {best_ba_solution}")
print(f"Best fitness from BA: {ba_model.g_best.target.fitness}")
print(f"Best real scheduling: {problem.decode_solution(best_ba_solution)}")
