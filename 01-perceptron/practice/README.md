
### Perceptron Implementation with NumPy
```python
import numpy as np

epsilon = 0.0000001

def perceptron (x1, x2):

    X = np.array([x1,x2])
    W = np.array([1.0,1.0])
    B = -1.5

    sum = np.dot(W,X) + B

    if sum > epsilon:
        return 1
    else:
        return 0

input1, input2 = [i for i in map(int, input().split())]
print(perceptron(input1, input2)
```
---

### Perceptron learning algorithm
```python
import numpy as np

epsilon = 0.0000001

X = np.array([ # x1, x2, bias
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([0,0,0,1]) 
W = np.zeros(len(X[0]) # X[0]=[0,0,1] ∴W=[0,0,0]

def step_func(t):
    if t > epsilon:
        return 1
    else:
        return 0

def perceptron_fit(X, Y, epochs=10): # epochs 안 주면 기본값이 10 / 현재는 epochs = 6
    global W # 함수 밖 W를 함수 안에서 수정하겠다
    eta = 0.2 # 학습률
    for t in range(epochs): # range(6) 반복할 때 t는 순서대로 t = 0,1,2,3,4,5
        print("==== epoch = ", t, " ====")
        for i in range(len(X)): # len(X)=4 ∴range(4), i=0,1,2,3
            predict = step_func(np.dot(X[i], W)) # 현재 입력 X[i]와 현재 가중치 W를 계산해서   
                                                 # 퍼셉트론이 예측한 결과를 predict에 저장한다
            error = Y[i] - predict
            W += eta * error * X[i]
            print("현재 처리 입력=", X[i], "정답=", Y[i], "출력=", predict, "변경된 가중치=", W)

perceptron_predict (X, Y):
    global W # perceptron_fit에서 학습된 W
    for x in X: # X 안에 있는 데이터를 x에 넣어라 x = [0,0,1], [0,1,1], [1,0,1], [1,1,1]
        print(x[0], x[1], " -> ", step_func(np.dot(x,W)))

perceptron_fit (X, y, 6)
perceptron_predict (X, y)
```
---

### Perceptron Implementation with Scikit-learn
```python
from sklearn.linear_model import Perceptron

X = [ [0,0], [0,1], [1,0], [1,1] ]
y = [ 0, 0, 0, 1 ]

clf = Perceptron ( tol = 1e-3, random_state=0 ) # Perceptron 객체 생성 후 clf 라는 변수에 저장

clf.fit ( X, y )

input1, input2 = [i for i in map(int, input().split())]
print ( int ( clf.predict ( [[ input1, input2 ]] ) [0] ) )
```
---

### Visualize the perceptron using matplotlib's pyplot
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.metrics import accuracy_score

import random
random.seed(32)  # 42 is an arbitrary number, you can use any number

# 퍼셉트론을 생성한다. tol는 종료 조건이다.
clf = Perceptron ( tol=1e-3 )

# 뭉쳐진 데이터를 만든다. 샘플의 개수는 총 100개, 클러스터의 개수는 2개이다.
X,y = make_blobs(n_samples=100, centers=2)
clf.fit(X,y)

print(accuracy_score(clf.predict(X),y))

# 데이터를 그래프 위에 표시한다.
plt.scatter(X[:,0], X[:,1], c=y, s=100)
plt.xlabel("x1")
plt.ylabel("x2")

# 데이터에서 최소 좌표와 최대 좌표를 계산한다.
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

# 0.1 간격으로 메쉬 그리드 좌표를 만든다.
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))

# 메쉬 그리드 데이터에 대하여 예측을 한다.
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)

# 컨투어를 그린다.
plt.contourf(xx, yy, Z, alpha=0.4)
plt.show()
```
