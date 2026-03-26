## Assignment 1 – OR Operation using a Perceptron  
Implement a perceptron model that performs the logical OR operation.  
The goal of training is to find weights and bias that correctly produce the OR output for all input combinations.  
```python
import numpy as np

def step_func(t): # 퍼셉트론 활성화 함수
  epsilon = 0.000001 # 부동소수점 오차 방지를 위함
  return 1 if t > epsilon else 0

def perceptron_fit(X, y, epochs=10): # 퍼셉트론 학습 알고리즘 구현
  global W # 학습 중 업데이트되는 가중치
  eta = 0.2 # 학습률

  for i in range(epoche): # 학습을 반복하는 파트
    print("== epoch =", t, "==")

    for i in range(len(X)):
        predict = step_func(np.dot(X[i], W)) # 퍼셉트론이 예측하는 값 계산
        error = y[i] - predict # 오차계산 (오차 = 실제 정답 - 퍼셉트론이 예측한 값)
        W += eta * error * X[i] # 가중치 업데이트
        print("현재 처리 입력 = ", X[i], "정답 = ", y[i], "출력 = ", predict, "변경된 가중치 = ", W)

def preceptron_predict(X, Y): # 최종 결과 확인 함수
  global W # 학습이 끝난 가중치

  for x in W:
      print(x[0], x[1], "->", step_func(np.dot(x,W)))

X = np.array([ # 훈련 데이터 셋
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([0, 1, 1, 1]) # OR 연산 정답 행렬
W = np.zeros(len(X[0])) # 가중치 저장 행렬

perceptron_fit(X, y, 4)
perceptron_predict(X, y)

```
<br><br>
---
<br><br>
## Assignment 2 – Perceptron Learning Process  
Show how the perceptron learning algorithm updates weights and bias using the training samples below.
Training Conditions
Learning rate η = 1.0
Initial parameters w1 = 0 w2 = 0 b = -1.0
Training Samples
Sample	x1	x2	Target
A	0	1	0
B	1	0	0
C	1	2	1
D	2	1	1
```python
import numpy as np

def step_func(t): # 퍼셉트론 활성화 함수
    epsilon = 0.0000001 # 부동소수점 오차 방지를 위함
    return 1 if t > epsilon else 0

def perceptron_fit(X, y, epochs=10): # 퍼셉트론 학습 알고리즘 구현
    global W # 학습 중 업데이트되는 가중치
    eta = 1.0 # 학습률은 1.0 으로 한다는 조건 만족

    for t in range(epochs): # 학습을 반복하는 파트
        print("==== epoch = ", t, "====")

        for i in range(len(X)):

            predict = step_func(np.dot(X[i], W)) # 퍼셉트론이 예측하는 값 계산
            error = y[i] - predict # 오차계산 (오차 = 실제 정답 - 퍼셉트론이 예측한 값)

            W += eta * error * X[i] # 가중치 업데이트

            print("현재 처리 입력=", X[i], "정답=", y[i], "출력=", predict, "변경된 가중치=", W)

def perceptron_predict(X, Y): # 최종 결과 확인 함수
    global W # 학습이 끝난 가중치
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x, W)))

X = np.array([ # 훈련 데이터 셋
    [0,1,1],
    [1,0,1],
    [1,2,1],
    [2,1,1]
])

y = np.array([0,0,1,1]) # 정답 행렬
W = np.array([0,0,-1.0]) # 가중치 저장 행렬
# 참고 : W == 가중치, W[2] == bias

perceptron_fit(X, y, 6)
perceptron_predict(X, y)

```
<br><br>
---
## Assignment 3 – Classification of 2D Data  
Using the 2D dataset shown in the figure, design a perceptron that separates the data into two different classes.  
Training Conditions
Learning rate η = 0.3
Initial bias b = -1.0
```python
import numpy as np

def step_func(t): # 퍼셉트론 활성화 함수
    epsilon = 0.0000001
    return 1 if t > epsilon else 0

def perceptron_fit(X, Y, epochs=10):  # 퍼셉트론 학습 알고리즘 구현
    global W # 학습 중 업데이트되는 가중치
    eta = 0.3 # 학습률은 0.3 으로 한다는 조건 만족

    for t in range(epochs): # 학습을 반복하는 파트
        print("==== epoch = ", t, "====")

        for i in range(len(X)):

            predict = step_func(np.dot(X[i], W)) # 퍼셉트론이 예측하는 값 계산
            error = Y[i] - predict # 오차계산 (오차 = 실제 정답 - 퍼셉트론이 예측한 값)

            W += eta * error * X[i]		# 가중치 업데이트

            print("현재 처리 입력=",X[i],"정답=",Y[i],"출력=",predict,"변경된 가중치=", W)

def perceptron_predict(X, Y): # 최종 결과 확인 함수
    global W # 학습이 끝난 가중치
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x, W)))

X = np.array([
    [0.1,0.7,1],			# 0 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
    [0.3,0.6,1],			# 0 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
    [0.4,0.2,1],			# 0 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
    [0.6,0.3,1],			# 0 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.

    [0.1,1,1],			  # 1 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
    [0.4,1,1],			  # 1 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
    [0.7,0.7,1],			# 1 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
    [0.9,0.4,1],      # 1 맨 끝의 1은 바이어스를 위한 입력 신호 1이다.
])

y = np.array([0,0,0,0,1,1,1,1]) # 정답 행렬
W = np.zeros(len(X[0])) # 가중치 저장 행렬
W[2] = -1.0 # W = [w1, w2, b] 구조인데, 바이어스의 초기값은 -1.0 이기 때문에

perceptron_fit(X, y, 20)
perceptron_predict(X, y)
```
<br><br>
