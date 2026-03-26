# Multi-Layer Perceptron (MLP)  
<br><br>
## 1. Multi-Layer Perceptron : MLP (다층 퍼셉트론)  
: 입력층과 출력층 사이에 은닉층(hidden layer)을 가지고 있는 퍼셉트론  

<br><br><br><br>
---  
<br><br><br><br>

## 2. Activation Function (활성화 함수)   
: 뉴런(노드)의 입력 신호의 합을 받아서 최종 출력값을 결정하는 함수  
• 신경망에서 선형 결과를 비선형(non-linear) 변환으로 바꿔주는 역할  
• 만약 활성화 함수가 없다면 신경망은 여러 층을 쌓아도 결국 하나의 선형모델과 동일해 짐  
• 선형 레이어는 많이 쌓아도 쓸모가 없다. 선형 레이어는 아무리 많아도 하나의 레이어로 대치될 수 있음  
• 다양한 활성화 함수가 사용될 수 있음  
<br><img width="600" height="254" alt="Image" src="https://github.com/user-attachments/assets/501a7113-a27a-49b0-b7f5-d7a671c49fc0" /><br>
<br><img width="60%" alt="Image" src="https://github.com/user-attachments/assets/05cf8a05-c820-40ee-9aaf-341833d0e341" /><br>

### ● Step Function (계단 함수)  
입력 신호의 총합이 0을 넘으면 1을 출력, 그렇지 않으면 0을 출력  
미분이 불가능하여 깊어지면 학습이 불가능  
0을 제외하고 모든 구간에서 gradient(기울기)가 0이므로 가중치 업데이트가 불가능함  
출력이 너무 단순하여 연속적인 확률표현이 불가능  
<br><img width="727" height="266" alt="Image" src="https://github.com/user-attachments/assets/63af9be5-87b5-4fe9-a40c-7a65363d32ab" /><br>
```python
import numpy as np
import matplotlib.pyplot ad plt # 그래프를 그리기 위한 라이브러리를 가져옴

def step(x) :
  result = x > 0.000001 # 크면 True, 작으면 False
  return result.astype(np.int32) # True는 1, False는 0

x = np.arange(-10.0, 10.0, 0.1) # -10.0 부터 10.0 까지 0.1 간격으로 숫자를 만들어서 x에 저장
y = step(x) # step 함수에 x를 넣고, x 안에 있는 값들이 0.000001보다 큰지 검사 후, True/False → 1/0 으로 바꾼 결과를 y에 저장
plt.plot(x, y) # x를 가로축, y를 세로축으로 해서 그래프를 그림
plt.show() # 만든 그래프를 화면에 보여줌

```
<br>  

### ● Sigmoid Function (시그모이드 함수)    
S자와 같은 형태    
전통적인 활성화 함수  
<br><img width="782" height="276" alt="Image" src="https://github.com/user-attachments/assets/81515b52-4ca0-465f-910a-1b976dbcf222" /><br>
```python
import numpy as np
import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리를 가져옴

def sigmoid(x) :
  return 1.0 / ( 1.0 + np.exp(-x) ) # np.exp(-x) → e^(-x)

x = np.arange(-10.0, 10.0, 0.1) # -10.0 부터 10.0 까지 0.1 간격으로 숫자를 만들어서 x에 저장
y = sigmoid(x) # sigmoid 함수에 x를 넣고, x 안에 있는 값들에 대해 계산이 일어난 후, 계산된 결과들이 y에 저장됨
               # sigmoid(-10) ≈ 0 / sigmoid(0) x = 0.5 / sigmoid(10) x ≈ 1
plt.plot(x, y) # x를 가로축, y를 세로축으로 해서 그래프를 그림
plt.show() # 만든 그래프를 화면에 보여줌
```
<br>

### ● tanh 함수  
시그모이드 함수와 아주 비슷하지만 출력값이 -1에서 1까지  
넘파이에서 제공하므로 별도의 함수 작성 불필요  
<br><img width="782" height="291" alt="Image" src="https://github.com/user-attachments/assets/2369cafb-afff-4b14-a6cd-cd4034aa47c7" /><br>
```python
import numpy as np
import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리를 가져옴

x = np.linspace ( -np.pi, np.pi, 60 ) # np.linspace → 일정한 간격으로 숫자를 여러 개 만드는 함수
                                      # np.pi → π ( 3.14 )
                                      # 즉, -3.14 부터 3.14 까지 같은 간격으로 총 60개의 값을 만들어서 x에 저장
y = np.tanh(x) # x 안에 있는 값 각각에 대해 tanh 값을 구해서 y에 저장함
               # tanh(0) = 0 / tanh(큰 양수) ≈ 1 / tanh(큰 음수) ≈ -1
plt.plot(x, y) # x를 가로축, y를 세로축으로 해서 그래프를 그림
plt.show() # 만든 그래프를 화면에 보여줌
```
###### arange(start, end, 간격)는 간격을 정함 <br> linspace(start, end, 개수)는 개수를 정함  
<br>


### ● Rectifed Linear Unit function (ReLU 함수)    
입력이 0을 넘으면 그대로 출력, 입력이 0보다 적으면 0을 출력      
최근에 가장 인기 있는 활성화 함수  
<br><img width="742" height="278" alt="Image" src="https://github.com/user-attachments/assets/33cbac22-8964-4dad-978e-e392848f8653" /><br>
```python
import numpy as np
import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리를 가져옴

def relu(x):
  return np.maximum(x, 0) # x 와 0 을 비교해서 더 큰 값을 고르는 함수
                          # x 가 0 보다 작으면 0, 0보다 크면 x
x = np.arange ( -10.0, 10.0, 0.1) # -10.0 부터 10.0 까지 0.1 간격으로 숫자를 만들어서 x에 저장
y = relu(x) # relu 함수에 x 를 넣고 그 결과를 y에 저장
plt.plot(x , y) # x를 가로축, y를 세로축으로 해서 그래프를 그림
plt.show # 만든 그래프를 화면에 보여줌

```

<br><br><br><br>
---  
<br><br><br><br>

## 3. MLP의 순방향 패스  
: 입력 신호가 입력층 유닛에 가해지고 이 입력 신호가 은닉층을 통해 출력층으로 전파되는 과정
<img width="583" height="437" alt="image" src="https://github.com/user-attachments/assets/2bae0fd4-a627-4c0a-9a86-bdb2e6758589" />  
손계산 Time~  
<img width="529" height="288" alt="image" src="https://github.com/user-attachments/assets/eeedae6b-2be6-4738-9ad8-fe9be6e9a4b5" />
<img width="327" height="162" alt="image" src="https://github.com/user-attachments/assets/9cf5d9c8-eb7a-4736-909f-e5340397dd25" />



















