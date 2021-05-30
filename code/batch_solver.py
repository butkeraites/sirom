import pandas as pd

class ProblemsBucket:
    'Class that will generate and storage all instances used to solve a problem like:'
    'min c * x : S.t. [lb_A, ub_A] * x - [lb_b, ub_b] <= 0'
    def __init__(self, c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_cenarios = 100):
        self.status = []
        self.coefficient = {}
        self.number_of_cenarios = -1
        self.__coefficient_validation(c_value,"objective")
        self.__coefficient_validation(lb_A_value,"lb_constraint")
        self.__coefficient_validation(ub_A_value,"ub_constraint")
        self.__coefficient_validation(lb_b_value,"lb_rhs")
        self.__coefficient_validation(ub_b_value,"ub_rhs")
        self.__number_of_cenarios_validation(number_of_cenarios)
        self.__dimension_validation()
        self.__problem_integrity_validation()
        self.__generate_all_coefficients()

    def __coefficient_validation(self, coefficient, identification):
        if coefficient :
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
            lb_constraint_equations, lb_constraint_coefficients = self.coefficient["lb_constraint"].shape
            ub_constraint_equations, ub_constraint_coefficients = self.coefficient["ub_constraint"].shape
            lb_rhs_coefficients, lb_rhs_dimension = self.coefficient["lb_rhs"].shape
            ub_rhs_coefficients, ub_rhs_dimension = self.coefficient["ub_rhs"].shape
            if  objective_coefficients     == lb_constraint_coefficients \
            and lb_constraint_coefficients == ub_constraint_coefficients \
            and lb_constraint_equations    == ub_constraint_equations \
            and ub_constraint_equations    == ub_rhs_coefficients \
            and lb_rhs_dimension           == ub_rhs_dimension \
            and lb_rhs_coefficients        == ub_rhs_coefficients \
            and objective_equations        == 1 \
            and ub_rhs_dimension           == 1:
                self.status.append("[OK] Optimization batch creation succeeded")
            else:
                self.status.append("[ERROR] Dimension inconsistency detected")

    def __number_of_cenarios_validation(self,number_of_cenarios):
        if number_of_cenarios:
            if isinstance(number_of_cenarios,int) and (number_of_cenarios >= 0):
                self.status.append("[OK] Successfuly acquired number of cenarios")
                self.number_of_cenarios = number_of_cenarios
            else:
                self.status.append("[ERROR] Failed acquiring number of cenarios")
        else:
            self.status.append("[ERROR] Undefined number of cenarios")

    def __problem_integrity_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Optimization batch creation failed")
        else:
            self.status.append("[OK] Optimization batch creation succeeded")
    
    def __generate_all_coefficients(self):
       #TODO: GENERATE ALL COEFFICIENTS 