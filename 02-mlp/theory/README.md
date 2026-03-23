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

&nbsp;  
### ● Step Function (계단 함수)  
입력 신호의 총합이 0을 넘으면 1을 출력, 그렇지 않으면 0을 출력  
미분이 불가능하여 깊어지면 학습이 불가능  
0을 제외하고 모든 구간에서 gradient(기울기)가 0이므로 가중치 업데이트가 불가능함  
출력이 너무 단순하여 연속적인 확률표현이 불가능  
<br><img width="727" height="266" alt="Image" src="https://github.com/user-attachments/assets/63af9be5-87b5-4fe9-a40c-7a65363d32ab" /><br><br>

### ● Sigmoid Function (시그모이드 함수)    
S자와 같은 형태    
전통적인 활성화 함수  
<br><img width="782" height="276" alt="Image" src="https://github.com/user-attachments/assets/81515b52-4ca0-465f-910a-1b976dbcf222" /><br><br>

### ● Rectifed Linear Unit function (ReLU 함수)    
입력이 0을 넘으면 그대로 출력, 입력이 0보다 적으면 0을 출력      
최근에 가장 인기 있는 활성화 함수  
<br><img width="742" height="278" alt="Image" src="https://github.com/user-attachments/assets/33cbac22-8964-4dad-978e-e392848f8653" /><br><br>

### tanh 함수  
시그모이드 함수와 아주 비슷하지만 출력값이 -1에서 1까지  
넘파이에서 제공하므로 별도의 함수 작성 불필요  
<br><img width="782" height="291" alt="Image" src="https://github.com/user-attachments/assets/2369cafb-afff-4b14-a6cd-cd4034aa47c7" /><br><br>


