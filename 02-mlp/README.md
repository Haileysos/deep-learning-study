# Multi-Layer Perceptron (MLP)  
### : 입력층과 출력층 사이에 은닉층(hidden layer)을 가지고 있는 퍼셉트론  

<br><br><br><br>

# Summary

입력층과 출력층 사이에 은닉층(hidden layer)을 가지고 있는 신경망을 다층퍼셉트론(multilayer perceptron: MLP)이라고 함

◆ MLP를 학습시키기 위하여 역전파 알고리즘(back-propagation)이 재발견되었다.  
이 알고리즘이 지금까지도 신경망 학습 알고리즘의 근간이되고 있다.  

◆ 역전파 알고리즘은 입력이 주어지면 순방향으로 계산하여 출력을 계산한 후에 실제 출력과 우리가 원하는 출력 간의 오차를 계산한다.  
이 오차를 역방향으로 전파하면서 오차를 줄이는 방향으로 가중치를 변경한다.  
