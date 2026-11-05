import numpy as np
import typing as tp
import numpy.typing as npt
from numpy.typing import NDArray

# Não esquecer da DocString!!! Incluir ao final
# Pessoal está implementando em ingles ou portugues?

def MSE(y_true, y_pred): #DAVI
    squared_errors = (y_true - y_pred) ** 2
    return float(np.mean(squared_errors))

def RMSE(): #DAVI
    pass

def MAE(): #DAVI
    pass

def RSS(): #DAVI
    pass

def R2(): #DAVI
    pass

def R2_ajustado(): #DAVI
    pass

def RSE(): #DAVI
    pass

