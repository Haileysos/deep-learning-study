# Gradient Descent (경사 하강법)  
: 손실함수의 기울기를 이용하여 손실값이 감소하는 방향으로 가중치(파라미터)를 반복적으로 업데이트하여 최소값을 찾는 최적화 방법

- 역전파 알고리즘은 신경망 학습 문제를 최적화 문제(optimization)로 접근함  
- 우리의 목표는 손실함수 값을 최소로 하는 가중치를 찾는 것  
<br><img width="30%" alt="image" src="https://github.com/user-attachments/assets/4f90a9a5-f067-4206-9609-15287916ca38" /><br>  
    - 기울기가 양수 → 오른쪽으로 갈수록 올라감  ∴ 왼쪽으로 가야 내려감 → 가중치를 감소시켜야 함  
    - 기울기가 음수 → 오른쪽으로 갈수록 내려감  ∴ 오른쪽으로 가야 내려감 → 가중치를 증가시켜야 함  

###### 손실함수를 가중치로 미분한 값 = 손실함수를 x로 미분한 값 = 기울기 y  
<br>  

## Example  
경사하강법을 사용하여 위 함수의 최소값을 찾는 파이썬 프로그램을 작성
<br><img width="50%" alt="image" src="https://github.com/user-attachments/assets/d1d300a1-2f96-48f3-aad7-e56c8ed6ef15" /><br>  

수학적으로는 이 함수가 `x = 3` 일 때, `기울기가 0` 이 되며, `최소값 y = 10` 을 갖는다는 것을 알 수 있음  

그러나 신경망에서는 이러한 값을 직접 구하지 않고 **`기울기`를 활용해** 손실함수 값`y`을 최소로 하는 가중치`x`를 찾아가야 함  

```python
x = 10 # 처음 시작할 값 
learning_rate = 0.2 # 학습률
precision = 0.00001 # 종료 기준 (이 코드에서는 사용 안 함)
max_iterations = 100 # 최대 반복 횟수

# 손실함수를 람다식으로 정의
loss_func = lambda x: (x-3)**2 + 10

# 그래디언트를 람다식으로 정의 (손실함수의 미분값)  
gradient = lambda x: 2 * x - 6

# 경사 강하법
for i in range(max_iterations):
  x = x - learning_rate * gradient(x) # x 값 업데이트
  print("손실함수값(", x, ") =", loss_func(x)) # 함수값 출력

print("최소값 = ", x)
```
<br>

## 손실함수 그래프의 시각화  

### ➡️ 3D

<br><img width="15%" alt="image" src="https://github.com/user-attachments/assets/9200c0c4-a47a-4ebf-8f2e-48d5daffc269" /><br>
```python
from mpl_toolkits.mplot3d import axes3d # 3차원 그래프 도구 불러옴
import matplotlib.pyplot as plt # 그래프 그리기 라이브러리
import numpy as np

x = np.arange(-5, 5, 0.5) # -5 부터 5 까지 0.5 씩 증가하는 숫자들
                          # x 축 값
y = np.arange(-5, 5, 0.5) # y 축 값
X, Y = np.meshgrid(x, y) # x, y 좌표 격자 생성
Z = X**2 + Y**2 # 넘파이 연산 (각 좌표에서의 함수값 계산)

fig = plt.figure(figsize=(6,6)) # 가로 6 세로 6 크기의 그래프 창 만듦
ax = fig.add_subplot(111, projection='3d') # 그래프 창 안에 그릴 3차원 좌표축(ax) 생성

# 3차원 그래프를 그림
ax.plot_surface(X, Y, Z)
plt.show() # 그래프 출력
```
<table>
  <tr>
    <td width="300">
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/d3b59ad6-daf2-4c4e-966e-f613ee85b619" />
    </td>
    <td>
      2차원 평면의 모든 좌표점에 대해 함수값을 계산한 뒤, 그것을 3차원 곡면으로 시각화함
    </td>
  </tr>
</table>
<br>

### ➡️ 화살표

<br><img width="15%" alt="image" src="https://github.com/user-attachments/assets/9200c0c4-a47a-4ebf-8f2e-48d5daffc269" />

<img width="27%" alt="image" src="https://github.com/user-attachments/assets/c5f8fbfa-b7ab-4eae-a876-fa72f246e017" /><br>

```python
import matplotlib.pyplot as plt # 그래프 그리기 라이브러리
import numpy as np

x = np.arange(-5, 5, 0.5) # -5 부터 5 까지 0.5 씩 증가하는 숫자들
                          # x 축 값
y = np.arange(-5, 5, 0.5) # y 축 값
X, Y = np.meshgrid(x, y) # x, y 좌표 격자 생성

# 원래 손실함수의 그래디언트는 (2x, 2y) 인데, 손실이 줄어드는 방향으로 가야하므로
# -1 을 곱하여 (-2x, -2y) 를 사용함
U = -2*X # x = 3 이면 U = -6 → 왼쪽으로 가라는 뜻 / x = -2 이면 U = 4 → 오른쪽으로 가라는 뜻
V = -2*Y # y = 4이면 V = -8 → 아래쪽으로 가라는 뜻 / y = -3이면 V = 6 → 위쪽으로 가라는 뜻
# 즉, 둘 다 (0,0), (0,0) 중심 쪽으로 향하게 됨

plt.figure() # 그래프 창 만
Q = plt.quiver(X, Y, U, V, units='width') # 화살표 벡터 그래프를 그리는 함수
                                          # X,Y → 화살표를 그릴 위치 / U,V → 화살표가 향하는 방향
                                          # 좌표(X,Y)마다 방향(U,V)을 가진 화살표를 그려줌
plt.show() # 그래프 출력
```
<table>
  <tr>
    <td width="300">
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/5e788f16-07b9-4385-8666-cdc7f886a986" />
    </td>
    <td>
      그래디언트에 -1을 곱하여, 그래디언트의 역방향 화살표를 시각화함 <br>
      화살표의 방향이 최소값(x=y=0) 을 가리키고 있음 <br>
      즉, 어떤 위치에서든지 그래디언트의 역방향으로 가면 최저값에 도달할 수 있음을 보여줌 <br><br>
      그래디언트는 함수가 가장 빠르게 증가하는 방향을 나타내므로, <br>
      손실함수를 최소화하기 위해 경사 하강법에서는 그래디언트의 반대방향(-gradient)으로 이동함
    </td>
  </tr>
</table>
<br>


