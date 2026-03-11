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

## 2. 퍼셉트론 구현

### 기본 Python 사용

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

### Numpy 사용

```python
import numpy as np

epsilon = 0.0000001 # 부동소수점 오차 방지를 위함

def perceptron(x1, x2): # 퍼셉트론이라는 작은 AI를 만듦
                        # 두 개의 숫자 를 입력으로 받음 x1=1, x2=0
    X = np.array([x1, x2]) # 입력을 배열로 만듦 X=[1,0]
    W = np.array([1.0, 1.0]) # 가중치를 나타냄 w1=1, w2=1
    B = -1.5 # 바이어스

    sum = np.dot(W, X) + B # 퍼셉트론 핵심 공식
                           # w1x1 + w2x2 + b
    if sum > epsilon:
        return 1
    else:
        return 0

print(perceptron(0,0)) # 0x1 + 0x1 + (-1.5) = -1.5 < epsilon -> 0
print(perceptron(1,0)) # 1x1 + 0x1 + (-1.5) = -0.5 < epsilon -> 0
print(perceptron(0,1)) # 0x1 + 1x1 + (-1.5) = -0.5 < epsilon -> 0
print(perceptron(1,1)) # 1x1 + 1x1 + (-1.5) = 0.5 > epsilon -> 1
```

&nbsp;
---
&nbsp;  

## 3. 퍼셉트론 학습 알고리즘  

퍼셉트론의 핵심은 학습을 통해 가중치를 자동으로 조정하는 것  

즉, 신경망이 데이터를 이용하여 적절한 가중치를 스스로 찾음  
&nbsp;
### 학습 알고리즘  
input : 학습 데이터 (x<sup>1</sup>, d<sup>1</sup>), (x<sup>2</sup>, d<sup>2</sup>), ..., (x<sup>m</sup>, d<sup>m</sup>)  

모든 가중치 `w`와 바이어스 `b`를 0또는 작은 난수로 초기화  
while (가중치 `w`가 변경되지 않을 때까지 반복)  
　  for 각 학습 데이터 x<sup>k</sup>와 정답 d<sup>k</sup>  
　　    y<sup>k</sup>(t) = f(w(t) x<sup>k</sup>)  
　　    모든 가중치 w<sub>i</sub>에 대하여 w<sub>i</sub> (t+1) = w<sub>i</sub> (t) + η (d<sup>k</sup> - y<sup>k</sup>(t)) x<sup>k</sup><sub>i</sub>  
    
1. 학습 데이터를 입력
2. 현재 가중치로 예측값을 계산한다.
3. 실제 정답과 비교하여 오차(error)를 계산
4. 오차를 이용하여 가중치를 수정
5. 이 과정을 여러 번 반복

가중치 업데이트 식

퍼셉트론의 가중치는 다음과 같이 업데이트된다.

w<sub>i</sub> (t+1) = w<sub>i</sub> (t) + η (d<sup>k</sup> - y<sup>k</sup>(t)) x<sup>k</sup><sub>i</sub>

`η` : 학습률 (learning rate)

`d` : 실제 정답 (desired output)

`y` : 예측값 (prediction)

`x<sub>i</sub>` : 입력값

`w<sub>i</sub>` : 가중치

퍼셉트론이 1을 0으로 잘못 식별했다고 하자. 가중치의 변화량은 η * (1-0) * x<sup>k</sup><sub>i</sub> 가 된다. 따라서 가중치는 증가된다. 가중치가 증가되면 출력도 증가되어서 출력이 0에서 1이 될 가능성이 있다.  

반대로 0을 1로 잘못 식별했다고 하자. 가중치의 변화량은 η * (0-1) * x<sup>k</sup><sub>i</sub> 가 된다. 따라서 가중치는 줄어든다. 가중치가 줄어들면 출력도 감소되어서 출력이 1에서 0이 될 가능성이 있다.  

&nbsp;
---
&nbsp;  

## 4. 퍼셉트론 학습 구현  

### Python 구현

```python
import numpy as np

epsilon = 0.0000001

def step_func(t): # 퍼셉트론 활성화 함수 (최종 판단 함수)
    if t > epsilon:
        return 1
    else:
        return 0

def perceptron_fit(X, Y, epochs=10): # 퍼셉트론 학습 알고리즘 구현
                                     # X:입력데이터/Y:정답데이터/epochs:몇번반복할지
    global W # 가중치 (함수 밖에 있는 가중치를 가지고 학습하며 수정함)
    eta = 0.2 # 학습

    for t in range(epochs): # 학습을 반복하는 파트 (epochs=10 이므로 10번 반복)
        print("==== epoch = ", t, "====") # 현재 몇 번째 반복 학습인지 보여
        for i in range(len(X)): # 현재 X는 4개의 입력이 있으므로 4번 돌림

            predict = step_func(np.dot(X[i], W)) # 퍼셉트론이 예측하는 값 계산
            error = Y[i] - predict # 오차계산 (오차 = 실제 정답 - 퍼셉트론이 예측한 값)

            W += eta * error * X[i] 

            print("현재 처리 입력=",X[i],"정답=",Y[i],"출력=",predict,"변경된 가중치=", W)
        print("================================")

def perceptron_predict(X, Y): # 최종 결과 확인 함수

    global W # 학습이 끝난 가중치 w

    for x in X:

        print(x[0], x[1], "->", step_func(np.dot(x, W)))


X = np.array([ # 훈련 데이터 셋
    [0,0,1], # x1=0, x2=0, bias=1
    [0,1,1], # x1=0, x2=1, bias=1
    [1,0,1], # x1=1, x2=0 bias=1
    [1,1,1] # x1=1, x2=1, bias=1
]) # 마지막 1은 바이어스를 위한 입력신호 1

y = np.array([0,0,0,1]) # 정답을 저장하는 넘파이 행렬
                        # 정답은 0, 0, 0, 1 -> 이 부분을 보고 이 코드가 AND 구나 하고 앎
W = np.zeros(len(X[0])) # 가중치를 저장하는 넘파이 행렬
                        # X[0] = [0,0,1] 이므로 len(X[0])=3, W=[0,0,0]
                        # 일단 처음에 아무것도 모르는 상태이기 때문에
perceptron_fit(X, y, 6) # 입력, 정답을 가지고 6번 반복 학습
perceptron_predict(X, y) # 학습 종료 후 최종 출력 확
```

### sklearn 구현

```python
from sklearn.linear_model import Perceptron

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]

clf = Perceptron(tol=1e-3, random_state=0)

clf.fit(X, y)

print(clf.predict(X))
```

&nbsp;
---
&nbsp;  

## 5. 퍼셉트론의 한계 (XOR)  

퍼셉트론은 선형 분류자(linear classifier)의 일종으로서  

데이터를 직선으로 분리할 수 있는 경우에만 학습이 가능    

하지만 XOR 문제는 직선 하나로 데이터를 분리할 수 없음  

XOR 논리 연산  

| x1 | x2 | y |
| -- | -- | - |
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

이 데이터는 한 개의 직선으로 분리할 수 없기 때문에  

단층 퍼셉트론(single-layer perceptron) 은 XOR 문제를 해결할 수 없다.  

&nbsp;
---
&nbsp;  

## 6. 다층 퍼셉트론 (Multi-layer Perceptron)

XOR 문제를 해결하기 위해 다층 퍼셉트론(Multi-layer Perceptron, MLP) 이 등장  

다층 퍼셉트론은 여러 개의 퍼셉트론을 층(layer) 구조로 연결한 신경망

구조

입력층 → 은닉층 → 출력층

은닉층(hidden layer)을 추가하면 데이터 공간을 변환할 수 있기 때문에  
단층 퍼셉트론으로 해결할 수 없던 문제도 해결할 수 있다.  

즉  

단층 퍼셉트론 → XOR 해결 불가  
다층 퍼셉트론 → XOR 해결 가능  

이 구조가 이후 딥러닝(Deep Learning) 의 기본 구조가 된다.








