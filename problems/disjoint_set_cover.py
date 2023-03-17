import os
import sys
import minorminer
import numpy as np
from multiprocessing import Pool
from dimod import BinaryQuadraticModel
from neal.sampler import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation

from utils.qubos import build_coef_matrix, to_qubo_coeffs
from utils.misc import dict_to_sample

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.append(repo_root)

class DisjointSetCover:
    def __init__(self, 
                 lambda_=1.1, 
                 sampler_type="qa",
                 decompose=True,
                 solver="Advantage_system4.1") -> None:

        self.default_sa_subproblem_reads = 100
        self.default_num_reads = 100
        self.default_qa_subproblem_reads = 2500
        self.default_qa_num_reads = 5000
        
        self.lambda_ = float(lambda_)
        self.sampler_type = sampler_type

        # initialize the sampler, integrating the specific quantum solver if needed
        assert self.sampler_type in ("sa", "qa")
        self.sampler_type = sampler_type
        self.decompose = decompose

        # connect to the AQC if using Quantum Annealing
        if self.sampler_type == "qa":
            assert solver in ("Advantage_system4.1",
                              "Advantage_system5.1", "DW_2000Q_6")
            if solver == "Advantage_system5.1":
                self.sampler = DWaveSampler(
                    region="eu-central-1", solver={"name": solver, "topology__type": 'pegasus'})
            elif solver == "Advantage_system4.1":
                self.sampler = DWaveSampler(
                    region="na-west-1", solver={"name": solver, 'topology__type': 'pegasus'})
            elif solver == "DW_2000Q_6":
                self.sampler = DWaveSampler(
                    region="na-west-1", solver={"name": solver, 'topology__type': 'chimera'})
            print("Solver: ", self.sampler.properties["chip_id"])

        elif self.sampler_type == "sa":
            self.sampler = SimulatedAnnealingSampler()


    def set_reads(self, num_reads):
        if num_reads is not None:
            return num_reads
        if self.sampler_type == "sa" and self.decompose:
            return self.default_sa_subproblem_reads
        if self.sampler_type == "sa" and not (self.decompose):
            return self.default_num_reads
        if self.sampler_type == "qa" and self.decompose:
            return self.default_qa_subproblem_reads
        if self.sampler_type == "qa" and not (self.decompose):
            return self.default_qa_num_reads


    def run(self, P, embedding=None, chain_strength=-1,
            chain_str_offset=0.5, num_reads=None,
            verbose=False, **anneal_params):
        """
            Solve the Disjoint Set Cover problem on the :param P: preference matrix.

            Params:
            - :param embedding: if provided, it is assumed to match the problem topology and will avoid computing
            a new embedding;  
            - :param chain_strength: chain strength value for Quantum Annealing. 
            If equal to -1, then chain strength will be set according to the Maximum Chain Length criterion on the problem embedding. 
            If None, then the default Uniform Torque compensation is used;  
            - :param chain_str_offset: value to be added to the computed chain strength 
            (only used with the MCL criterion);  
            - :param num_reads: same as num_reads for DWave Samplers' "sample" method;  
            - :params **anneal params: additional annealing parameters, following the DWave APIs at 
            (https://docs.dwavesys.com/docs/latest/c_solver_parameters.html)

            Returns:  
            - a binary vector z, encoding the selected columns of P.
        """

        N, M = P.shape
        ones_n = np.ones(shape=(N,))
        ones_m = np.ones(shape=(M,))

        # define the matrices of our formulation, embedding the constraints
        Q_tilde = self.lambda_ * np.matmul(P.transpose(), P)
        s_tilde = ones_m - 2*self.lambda_*np.matmul(P.transpose(), ones_n)

        # build the coefficients matrix:
        # - The upper triangular portion will contain couplings coming from constraints;
        # - the diagonal will contain the sum of the problem biases and the biases coming from constraints;
        # NOTE: Without constraints, the formulation does NOT include quadratic terms
        upper_diag = build_coef_matrix(Q_tilde, s_tilde)

        # transform the matrix to a python dictionary, feasible for Dwave solvers
        qubo = to_qubo_coeffs(upper_diag)
        bqm = BinaryQuadraticModel.from_qubo(qubo)

        # look for the embedding and set the chain strength in case of QA
        sampler = self.sampler
        left_aside_variables = []
        if self.sampler_type == "qa":
            # find the embedding (i.e. mapping) between the current logical graph (the bqm) and the qpu topology
            if embedding is None:
                src_graph = list(bqm.quadratic.keys())
                nums = []
                for num1, num2 in src_graph:
                    nums.append(num1)
                    nums.append(num2)
                left_aside_variables = set(bqm.variables) - set(nums)
                for v in left_aside_variables:
                    bqm.remove_variable(v)
                target_graph = self.sampler.edgelist
                embedding, found = minorminer.find_embedding(src_graph, target_graph, return_overlap=True)
                if not found: return np.ones(M)

            # choose the chain strength according to the maximum chain length criterion
            if chain_strength == -1:
                chain_strength = max(
                    list(len(chain) for chain in embedding.values())) + chain_str_offset
            elif chain_strength is None:
                chain_strength = uniform_torque_compensation(bqm, embedding)
            if verbose:
                print("Running with chain strength: ({}x{})".format(
                    N, M), chain_strength)

            # instantiate the sampler and run it to find low energy states of the possible instantiations
            sampler = FixedEmbeddingComposite(self.sampler, embedding=embedding)

        sampleset = sampler.sample(bqm, num_reads=self.set_reads(num_reads), chain_strength=chain_strength,
                                   return_embedding=True, **anneal_params)

        # compute the lowest energy sample from the returned sampleset
        first = sampleset.first
        tentative_solution = dict_to_sample(first.sample)

        # merge the returned solution with the pre-determined qubits
        if self.sampler_type == "qa":
            lowest_energy_sample = []
            added = 0
            for i in range(M):
                if i in left_aside_variables:
                    lowest_energy_sample.append(1)
                    added += 1
                else:
                    lowest_energy_sample.append(tentative_solution[i-added])
            lowest_energy_sample = np.array(
                lowest_energy_sample, dtype=np.int16)
        else:
            lowest_energy_sample = tentative_solution

        return lowest_energy_sample


    def wrapped_run(self, P, kwargs_dict):
        return self.run(P, **kwargs_dict)


    def submatrices(self, P, subproblem_size, return_last=False):
        # slice the preference matrix along the column axis
        prev_j = 0
        j = subproblem_size
        ret = []
        while prev_j < P.shape[1]:
            if j > P.shape[1]:
                submatrix = P[:, prev_j:]
            else:
                submatrix = P[:, prev_j:j]
            ret.append(submatrix.copy())
            prev_j = j
            j += subproblem_size

        if not return_last:
            last_submatrix = ret[-1]
            if last_submatrix.shape[1] < subproblem_size:
                ret = ret[:-1]
        return ret


    def divide_et_impera(self, P, subproblem_size=40, parallel=False, verbose=False, **kwargs):
        if parallel:
            pool = Pool()
        reads_per_subproblem = self.set_reads(kwargs.get("num_reads", None))

        total_reads = 0
        run_kwargs = {k: v for k, v in kwargs.items()}
        problem_dim = P.shape[1]
        indices = list(range(P.shape[1]))

        while P.shape[1] > subproblem_size:
            # divide
            subproblems = self.submatrices(P, subproblem_size)
            total_reads += reads_per_subproblem * len(subproblems)
            if verbose:
                log_str = f"Problem was divided into {len(subproblems)} subproblems. "
                log_str += f"Solving each with {reads_per_subproblem} reads. Total reads: {total_reads}"
                print(log_str)

            # impera
            if parallel:
                solutions = pool.starmap(self.wrapped_run, [(p, run_kwargs) for p in subproblems])
            else:
                solutions = []
                for p in subproblems:
                    sol = self.run(p, **run_kwargs)
                    solutions.append(sol)

            # remove unselected global indices
            removed = 0
            for i, sol in enumerate(solutions):
                offset = i*subproblem_size
                for j, item in enumerate(sol):
                    if item == 0:
                        rm_idx = j + offset - removed
                        indices.pop(rm_idx)
                        removed += 1

            # remove unselected columns
            partial_sol = []
            for sol in solutions:
                partial_sol.extend(sol)
            partial_sol = np.array(partial_sol)
            P = np.delete(P, obj=np.where(partial_sol == 0)[0], axis=1)

        # last execution if acceptable size reached (or first if problem was already small enough)
        sol = self.run(P, **run_kwargs)
        total_reads += reads_per_subproblem
        removed = 0
        for j, item in enumerate(sol):
            if item == 0:
                rm_idx = j - removed
                indices.pop(rm_idx)
                removed += 1

        # build the binary solution vector
        solution = np.zeros(shape=(problem_dim,), dtype=np.int8)
        solution[indices] = 1

        if verbose:
            print(f"Problem with {problem_dim} qubits solved with {total_reads} reads.")
        if parallel:
            pool.close()
        return solution


    def __call__(self, *args, **kwds):
        if self.decompose:
            return self.divide_et_impera(*args, **kwds)
        return self.run(*args, **kwds)
