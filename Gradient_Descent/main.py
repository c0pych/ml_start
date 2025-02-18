import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.


# необходимые функции
def scalar_factor(w, si):
    return sum(map(lambda x, y: x * y, w, si))

def create_si(x):
    return [1, x, x**2, x**3]

def a(w , x):
    return scalar_factor(w, create_si(x))

def dQ_func(w, x, sz): #градиент(вектор) показывающий направление движения в пространстве весов
    result_w = [0, 0, 0, 0]
    for i in range (sz):
        deviation = a(w, x[i])-func(x[i])
        derivative_at_a_point = list(map(lambda x: deviation *x,create_si(x[i])))
        result_w = list(map(lambda x, y: x+y , derivative_at_a_point , result_w))
    result_w = list(map(lambda x: x*2/sz , result_w))
    return result_w

def Q_func(w, x, sz): #ошибка
    result = 0
    for i in range (sz):
        deviation = a(w, x[i])-func(x[i])
        deviation = pow(deviation, 2)
        result+= deviation
    result = result/sz
    return result

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма

# собираем вместе
for j in range (200):
    for i in range (4):
        w[i] = w[i] - eta[i]*dQ_func(w,coord_x,sz)[i] 
Q = Q_func(w, coord_x, sz)        
