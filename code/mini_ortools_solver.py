from __future__ import print_function
from ortools.linear_solver import pywraplp

class MiniOrtoolsSolver:
    'Class that translate a Optimization problem to Ortools framework, solve and retrieve solving status'
    def __init__(self, optimization_problem, solver_selection = 'SCIP'):
        self.status = []
        self.problem = optimization_problem
        self.solver_selected = solver_selection
        self.__start_optimization_chain()
        
    def __start_optimization_chain(self):
        if any("[ERROR]" in status for status in self.problem.status):
            self.status.append("[ERROR] Optimization problem creation failed")
        else:
            self.status.append("[OK] Optimization problem creation succeeded")
            self.__create_solver()
    
    def __create_solver(self):
        self.solver = pywraplp.Solver.CreateSolver(self.solver_selected)
        if self.solver is None:
            self.status.append("[ERROR] Solver creation failed")
        else:
            self.status.append("[OK] Solver creation succeeded")
            self.__create_variables()

    def __create_variables(self):
        n_var = len(self.problem.coeficient['objective'])
        self.variables = [self.solver.NumVar(0, self.solver.infinity(), "x{}".format(str(id))) for id in range(n_var)]
        self.status.append("[OK] Variables creation succeeded")
        self.status.append("[INFO] Number of variables: {}".format(self.solver.NumConstraints()))
        self.__create_constraints()

    def __create_constraints(self):
        self.constraints = []
        
        self.status.append("[OK] Constraints creation succeeded")