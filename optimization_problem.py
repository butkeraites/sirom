import pandas as pd
class OptimizationProblem:
    'Class that will be served by http request to Optimization Workers. We are going to solve a problem of:'
    ' min c * x : S.t. A * x <= b'
    def __init__(self, c_value, A_value, b_value):
        self.status = []
        self.coeficient_validation(c_value, A_value, b_value)
        self.problem_integrity_validation()

    def coeficient_validation(self, c_value, A_value, b_value):
        if c_value :
            self.objective_coeficient = c_value
            self.status.append("[OK] Successfuly acquired objective coeficient")
        else:
            self.status.append("[ERROR] Undefined objective coeficient")

    def problem_integrity_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Optimization problem creation failed")
        else:
            self.status.append("[OK] Optimization problem creation succeeded")