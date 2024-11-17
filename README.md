# 한밭대학교 컴퓨터공학과 놀란바나나팀

**팀 구성**
- 20191742 유태원 
- 20191778 신승민
- 20181600 정하륜
- 20181595 이수호

## <u>Teamate</u> Project Background
- ### 필요성
  - 현대 사회에서는 가상현실(VR) 기술이 게임, 교육, 의료, 제조 등 다양한 분야에서 활용되며 점차 그 중요성이 증가하고 있습니다. 이러한 기술의 발전과 함께, 사용자가 자연스럽게 가상현실 환경과 상호작용할 수 있는 직관적이고 효율적인 입력 장치에 대한 요구가 증가하고 있습니다. 기존의 VR 컨트롤러는 물리적 장치를 손에 쥐고 사용하는 방식이 주를 이루며, 이는 동작의 자연스러움과 직관성을 저해할 뿐만 아니라, 특정 사용자(예: 장애인)에게는 사용이 어려운 경우도 많습니다. 근전도(EMG) 신호를 활용한 VR 컨트롤러로 이러한 문제를 해결하고 사용자 경험을 크게 향상시킬 혁신적인 시스템을 구현할 필요가 있습니다. 이를 통해 사용자는 더욱 자연스럽고 직관적인 방식으로 가상현실 환경과 상호작용할 수 있을 것입니다.
- ### 기존 해결책의 문제점
  - 물리적 컨트롤러의 제약
    - 기존 VR 컨트롤러는 사용자 손에 물리적으로 장착되거나 쥐어야 하므로 장시간 사용 시 피로감이 크고, 특정 동작 수행에 제한이 있습니다.
  - 유연성 부족
    - 일반적인 VR 컨트롤러는 제한된 동작만을 인식할 수 있어 다양한 인터랙션을 지원하는 데 한계가 있습니다.
  - 특수 사용자 그룹의 접근성 문제
    - 장애인 등 물리적 장치를 잡기 어려운 사용자들은 기존 컨트롤러를 사용하기 힘든 경우가 많아 보조 기기와의 통합이 필요합니다.
  
## System Design
  - ### System Requirements
    - 근전도 센서
      - 손이나 팔에 부착하여 근전도 신호를 수집할 수 있는 다채널 EMG 센서.
    - 데이터 처리 및 머신러닝 모델
      - 근전도 신호로부터 특징을 추출하고 이를 손동작으로 분류하는 머신러닝 모델.
    - 가상현실 환경
      - Unity 기반의 3D 가상현실 구현 환경으로, 입력 신호에 따라 가상 동작이 매끄럽게 연동될 수 있어야 함.
    
## Conclusion
  - 근전도 신호 기반 VR 컨트롤러는 사용자의 손동작을 자연스럽고 직관적으로 가상현실 환경과 연결하여 사용자 경험을 혁신적으로 개선할 수 있습니다. 기존 물리적 컨트롤러의 한계를 극복하며, 다양한 응용 분야에서 활용 가능성이 매우 높습니다. 또한, 본 시스템은 장애인을 포함한 다양한 사용자 그룹에게 접근성을 제공하여 기술의 포용성을 높이는 데 기여할 수 있습니다. 본 연구를 통해 VR 기술의 실용성을 높이고, 가상현실과 인간 간의 상호작용 가능성을 확장하는 데 큰 기여를 할 것으로 기대됩니다.

