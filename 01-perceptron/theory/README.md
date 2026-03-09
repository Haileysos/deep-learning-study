# Perceptron
&nbsp;
## 1. 퍼셉트론 (Perceptron)

### 퍼셉트론 : 1957년 Frank Rosenblatt가 고안한 인공 신경망 모델  

- 여러 입력정보를 이용해 두 가지 클래스( 1 or 0 ) 중 하나를 선택하는 가장 단순한 분류 모델
  
### 퍼셉트론 수식  

입력값과 가중치를 곱한 뒤 그 합들이 특정 임계값을 넘으면 1을 출력, 그렇지 않으면 0을 출력  

y = 1　　if (w1x1 + w2x2 ≥ T)   
　= 0　　otherwise    
 
- `x` : 입력값
- `w` : 가중치
- `T` : 임계값 (threshold)
- `b` : bias (b = -T)
&nbsp;
---
&nbsp;
## 2. 퍼셉트론 구현 (Python)

### 기본 Python 구현

```python
epsilon = 0.0000001

def perceptron(x1, x2):
    w1, w2, b = 1.0, 1.0, -1.5
    sum = x1*w1 + x2*w2 + b

    if sum > epsilon:
        return 1
    else:
        return 0

print(perceptron(0,0))
print(perceptron(1,0))
print(perceptron(0,1))
print(perceptron(1,1))
```
출력결과 0 0 0 1  
이는 AND 논리 연산과 동일한 결과이다.
| x1 | x2 | y |
| -- | -- | - |
| 0  | 0  | 0 |
| 1  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 1  | 1 |  

### Numpy를 이용해 구현

```python
import numpy as np

epsilon = 0.0000001

def perceptron(x1, x2):
    X = np.array([x1, x2])
    W = np.array([1.0, 1.0])
    B = -1.5

    sum = np.dot(W, X) + B 

    if sum > epsilon:
        return 1
    else:
        return 0

print(perceptron(0,0))
print(perceptron(1,0))
print(perceptron(0,1))
print(perceptron(1,1))
```

## 3. 퍼셉트론 학습 알고리즘

퍼셉트론의 핵심은 학습을 통해 가중치를 자동으로 조정하는 것이다.
즉, 신경망이 데이터를 이용하여 적절한 가중치(weight) 를 스스로 찾는다.

학습 알고리즘
input : 학습 데이터 (x<sup>1</sup>, d<sup>1</sup>), ..., (x<sup>2</sup>, d<sup>2</sup>), (x<sup>m</sup>, d<sup>m</sup>)
모든 가중치 `w`와 바이어스 `b`를 0또는 작은 난수로 초기화
while (가중치 `w`가 변경되지 않을 때까지 반복)
  for 각 학습 데이터 x<sup>k</sup>와 정답 d<sup>k</sup>
    y<sup>k</sup>(t) = f(w(t) x<sup>k</sup>)
    모든 가중치 w<sub>i</sub>에 대하여 w<sub>i</sub>(t+1) = w<sub>i</sub>(t) + η (d<sup>k</sup> - y<sup>k</sup>(t)) x<sup>k</sup><sub>i</sub>
    
1. 학습 데이터를 입력
2. 현재 가중치로 예측값을 계산한다.
3. 실제 정답과 비교하여 오차(error)를 계산
4. 오차를 이용하여 가중치를 수정
5. 이 과정을 여러 번 반복

가중치 업데이트 식

퍼셉트론의 가중치는 다음과 같이 업데이트된다.

w_i(t+1) = w_i(t) + η (d - y) x_i

여기서

`η` : 학습률 (learning rate)

`d` : 실제 정답 (desired output)

`y` : 예측값 (prediction)

`x<sub>i</sub>` : 입력값

`w<sub>i</sub>` : 가중치

퍼셉트론이 1을 0으로 잘못 식별했다고 하자. 가중치의 변화량은 η * (1-0) * xi k 가 된다. 따라서가중치는 증가된다. 가중치가 증가되면 출력도 증가되어서 출력이 0에서 1이 될 가능성이 있다.

반대로 0을 1로 잘못 식별했다고 하자. 가중치의 변화량은 η * (0-1) * xik 가 된다. 따라서 가중치는
줄어든다. 가중치가 줄어들면 출력도 감소되어서 출력이 1에서 0이 될 가능성이 있다.

## 4. 퍼셉트론 학습 구현 (Python)

```python
import numpy as np

epsilon = 0.0000001

def step_func(t):
    if t > epsilon:
        return 1
    else:
        return 0

X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([0,0,0,1])

W = np.zeros(len(X[0]))

def perceptron_fit(X, Y, epochs=10):

    global W
    eta = 0.2

    for t in range(epochs):
        print("epoch=", t, "======================")
        for i in range(len(X)):

            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict

            W += eta * error * X[i]

            print("현재 처리 입력=",X[i],"정답=",Y[i],"출력=",predict,"변경된 가중치=", W)
        print("================================")

def perceptron_predict(X, Y): 

    global W

    for x in X:

        print(x[0], x[1], "->", step_func(np.dot(x, W)))

perceptron_fit(X, y, 6)
perceptron_predict(X, y)
```


