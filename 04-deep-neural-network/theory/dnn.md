# DNN: Deep Neural Networks 심층 신경망  
: MLP(다층 퍼셉트론)에서 은닉층의 개수를 증가시킨 것  
<br><img width="50%" alt="image" src="https://github.com/user-attachments/assets/55b576a7-5581-4cf8-90bc-cb19ad2fb243" /><br>
<br><br><br>  
  
## MLP 의 문제점 해결  
### MLP의 문제점  
- 은닉층이 많아지면 출력층에서 계산된 그래디언트가 역전파되다가 값이 점점 작아져서 없어지는 문제점 발생
- 훈련 데이터가 충분하지 못하면, 과잉 적합(over fitting)이 될 가능성도 높아짐
 
### GPU의 도움  
- DNN의 학습 속도는 상당히 느리고 계산 집약적이기 때문에 학습에 시간과 자원이 많이 소모됨
- 최근 GPU 기술이 엄청나게 발전하면서 GPU가 제공하는 데이터 처리 기능을 딥러닝도 사용할 수 있게 됨
- 딥러닝 혁명에는 게이머들의 도움도 컸음

### 은닉층의 역할  
- 여러 개의 은닉층 중에서, 앞 단은 경계선(엣지)과 같은 저급 특징을, 뒷 단은 코너와 같은 고급 특징들을 추출함

### MLP vs DNN  
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/9d19af74-a9fa-44ef-b430-4fc5baac94b1" /><br>
