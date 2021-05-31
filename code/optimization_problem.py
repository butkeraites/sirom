import pandas as pd
import numpy as np

class OptimizationProblem:
    'Class that will be served by http request to Optimization Workers. We are going to solve a problem of:'
    'min c * x : S.t. A * x - b <= 0'
    def __init__(self, c_value, A_value, b_value):
        self.status = []
        self.coefficient = {}
        self.__coefficient_validation(c_value,"objective")
        self.__coefficient_validation(A_value,"constraint")
        self.__coefficient_validation(b_value,"rhs")
        self.__dimension_validation()
        self.__problem_integrity_validation()

    def __coefficient_validation(self, coefficient, identification):
        if isinstance(coefficient,(list,pd.core.series.Series,np.ndarray)) :
            try:
                self.coefficient[identification] = pd.DataFrame(coefficient)
                self.status.append("[OK] Successfuly acquired {} coefficient".format(identification))
            except:
                self.status.append("[ERROR] Failed acquiring {} coefficient".format(identification))
        else:
            self.status.append("[ERROR] Undefined {} coefficient".format(identification))

    def __dimension_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Dimension can not be evaluated")
        else:
            objective_coefficients, objective_equations = self.coefficient["objective"].shape
            constraint_equations, constraint_coefficients = self.coefficient["constraint"].shape
            rhs_coefficients, rhs_dimension = self.coefficient["rhs"].shape
            if  objective_coefficients   == constraint_coefficients \
            and constraint_equations    == rhs_coefficients \
            and objective_equations     == 1 \
            and rhs_dimension           == 1:
                self.status.append("[OK] Optimization problem creation succeeded")
            else:
                self.status.append("[ERROR] Dimension inconsistency detected")

    def __problem_integrity_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Optimization problem creation failed")
        else:
            self.status.append("[OK] Optimization problem creation succeeded")