import numpy as np
import typing as tp
import numpy.typing as npt
from numpy.typing import NDArray

# Não esquecer da DocString!!! Incluir ao final
# Pessoal está implementando em ingles ou portugues?

#PRECISO GARANTIR ALGUMAS VALIDAÇÕES DE ENTRADA, COMO MESMO TAMANHO, NÃO VAZIO E ETC!!! ACHO QUE ESSAS
#VALIDAÇÕES DEVEM SE REPETIR PARA AS DEMAIS MÉTRICAS... TALVEZ SEJA INTERESSANTE SEGMENTAR ESSES TESTES 


def MSE(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float: #DAVI
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (NDArray[np.float64]): The true values.
    y_pred (NDArray[np.float64]): The predicted values.

    Returns:
    float: The Mean Squared Error.
    """

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

