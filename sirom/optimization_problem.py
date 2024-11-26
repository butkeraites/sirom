from typing import List
import pandas as pd
import numpy as np


class Coefficient:
    def __init__(self, c, A, b):
        self._objective = c
        self._constraint = A
        self._rhs = b

    @property
    def objective(self):
        return self._objective

    @property
    def constraint(self):
        return self._constraint

    @property
    def rhs(self):
        return self._rhs


class OptimizationProblem:
    "Class that will be served by http request to Optimization Workers. We are going to solve a problem of:"
    "min c * x : S.t. A * x - b <= 0"

    def __init__(
        self,
        c_value: np.ndarray,
        A_value: np.matrix[np.dtype, np.dtype],
        b_value: np.ndarray,
    ):
        self.status: List[str] = []
        c_validated = self.__coefficient_validation(c_value, "objective")
        A_validated = self.__coefficient_validation(A_value, "constraint")
        b_validated = self.__coefficient_validation(b_value, "rhs")
        self.__set_coefficient(c_validated, A_validated, b_validated)
        self.__dimension_validation()
        self.__problem_integrity_validation()

    def __set_coefficient(self, c_validated, A_validated, b_validated):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Coefficient cannot be defined")
            return
        self.coefficient = Coefficient(c_validated, A_validated, b_validated)

    def __coefficient_validation(
        self, coefficient: np.ndarray | np.matrix, identification: str
    ) -> pd.DataFrame:
        if not isinstance(coefficient, (np.matrix, np.ndarray)):
            self.status.append(
                "[ERROR] Undefined {} coefficient".format(identification)
            )
        try:
            return pd.DataFrame(coefficient)
        except:
            self.status.append(
                "[ERROR] Failed acquiring {} coefficient".format(identification)
            )
            return pd.DataFrame()
        self.status.append(
            "[OK] Successfuly acquired {} coefficient".format(identification)
        )

    def __dimension_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Dimension can not be evaluated")
            return
        objective_coefficients, objective_equations = self.coefficient.objective.shape
        (
            constraint_equations,
            constraint_coefficients,
        ) = self.coefficient.constraint.shape
        rhs_coefficients, rhs_dimension = self.coefficient.rhs.shape
        if not (
            objective_coefficients == constraint_coefficients
            and constraint_equations == rhs_coefficients
            and objective_equations == 1
            and rhs_dimension == 1
        ):
            self.status.append("[ERROR] Dimension inconsistency detected")
            return
        self.status.append("[OK] Optimization problem creation succeeded")

    def __problem_integrity_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Optimization problem creation failed")
            return
        self.status.append("[OK] Optimization problem creation succeeded")
