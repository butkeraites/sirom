import pandas as pd
class OptimizationProblem:
    'Class that will be served by http request to Optimization Workers. We are going to solve a problem of:'
    ' min c * x : S.t. A * x <= b'
    def __init__(self, c_value, A_value, b_value):
        self.status = []
        self.coeficient = {}
        self.__coeficient_validation(c_value,"objective")
        self.__coeficient_validation(A_value,"constraint")
        self.__coeficient_validation(b_value,"rhs")
        self.__dimension_validation()
        self.__problem_integrity_validation()

    def __coeficient_validation(self, coeficient, identification):
        if coeficient :
            try:
                self.coeficient[identification] = pd.DataFrame(coeficient)
                self.status.append("[OK] Successfuly acquired {} coeficient".format(identification))
            except:
                self.status.append("[ERROR] Failed acquiring {} coeficient".format(identification))
        else:
            self.status.append("[ERROR] Undefined {} coeficient".format(identification))

    def __dimension_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Dimension can not be evaluated")
        else:
            objective_coeficients, objective_equations = self.coeficient["objective"].shape
            constraint_equations, constraint_coeficients = self.coeficient["constraint"].shape
            rhs_coeficients, rhs_dimension = self.coeficient["rhs"].shape
            if  objective_coeficients   == constraint_coeficients \
            and constraint_equations    == rhs_coeficients \
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