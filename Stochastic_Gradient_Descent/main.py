import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


# здесь объявляйте необходимые функции
def scalar_factor(w, si): #скалярное умножение
    return sum(map(lambda x, y: x * y, w, si))

def create_si(x): #задание признакого вектора
    return [1, x, x**2, x**3]

def a(w , x): #предиктивное значение
    return scalar_factor(w, create_si(x))

def dQk_func(w, x, k,  K): #подсчет градиента
    derivative_at_a_point = np.array([0., 0., 0., 0.])
    for i in range (k, k+K): 
        deviation = (a(w, x[i])-func(x[i]))
        derivative_at_a_point += np.array(list(map(lambda x: deviation*x,create_si(x[i]))))
        
    derivative_at_a_point = derivative_at_a_point*2/K    
    return derivative_at_a_point

def L_func(w, x):
    deviation = (a(w, x)-func(x))**2
    return deviation

def Qk_func(w, x, k, K):
    result = 0
    for i in range (k, k+K):
        deviation = L_func(w, x[i])
        result+= deviation
    result = result/K
    return result 

def Q_func(w, x,sz):
    result = 0
    for i in range (sz):
        deviation = L_func(w, x[i])
        result+= deviation
    result = result/sz
    return result 


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 50 # размер мини-батча (величина K = 50)

Qe = Q_func(w, coord_x, sz)  #начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел


for i in range (N):
    k = np.random.randint(0, sz-batch_size)
    dQk = dQk_func(w, coord_x, k, batch_size)
    w = w - eta*dQk
    Qe = lm*Qk_func(w, coord_x, k, batch_size ) + (1-lm)*Qe
Q = Q_func(w, coord_x, sz)
