from src.util.util import proj_root_dir

from src.method.simple_exponential_smoothing import run_method as run_method1
from src.method.double_exponential_smoothing import run_method as run_method2
from src.method.triple_exponential_smoothing import run_method as run_method3

from src.method.arima import run_method as run_method4
from src.method.sarima import run_method as run_method5
from src.method.sarimax import run_method as run_method6

from src.method.ml_LightGB import run_method as run_method7
from src.method.ml_RandomForest import run_method as run_method8
from src.method.ml_LinearReg import run_method as run_method9


def run():
    pass

    # simple_exponential_smoothing, double_exponential_smoothing, triple_exponential_smoothing
    run_method1()
    # run_method2()
    # run_method3()
    #
    # arima, sarima, sarimax
    # run_method4()
    # run_method5()
    # run_method6()
    #
    # ml_LightGB, ml_RandomForest, ml_LinearReg
    # run_method7()
    # run_method8()
    # run_method9()


def main():
    run()


if __name__ == '__main__':
    main()
