## 1. Data Extension
기존의 Casual Video(Free dataset) 단계를 넘어, 실제 자율주행 및 실세계 이동 로봇 환경으로 데이터 도메인을 확장하여 모델의 범용성을 입증합니다.
- **자율주행 데이터셋 (Waymo / KITTI / NuScenes) 연계:**
    - **구체화:** 자율주행 차량에 탑재된 멀티 카메라 환경(Multi-view) 또는 전방 카메라 시퀀스 영상을 활용합니다.
    - **목적:** 자율주행 환경은 카메라가 일직선 혹은 완만한 곡선으로 전진하는 독특한 궤적(Trajectory)을 가집니다. Unposed 3DGS 파이프라인이 이러한 주행 환경에서도 눈송이에 방해받지 않고 도로(Road Layout)와 주변 구조물을 정확하게 3D Recon 해낼 수 있는지를 체크합니다.

## 2. Weather Intensity Control
단순히 한두 가지 형태의 눈을 입히는 것을 넘어, 날씨의 강도를 조절하여 모델의 한계(Limit)를 테스트하는 정밀 실험 환경을 구축합니다.
- **Text-to-Video Diffusion 및 4D 제어 모델 활용:**
    - **구체화:** 최근 공개된 비디오 생성 AI(예: Sora, Runway Gen-3) 또는 가우시안 기반 날씨 편집 모델(WeatherEdit) 에 프롬프트를 다르게 주어 강설량을 세분화합니다.
    - 이전 연구에서 다루지 못한 부분을 확장하기 위함입니다.

## 3. Weather Heatmap Loss New Model for LongSplat
이전 연구 진행하며 떠오른 아이디어를 Claude와 함께 구체화해 보았습니다.
- **MWFormer 기반의 가벼운 Heatmap Extractor (Freeze):**
    - **구체화:** MWFormer의 엔코더 단을 활용해, 입력 비디오 프레임에서 날씨 아티팩트의 물리적 강도와 위치를 나타내는 $H_t$ 지도 레이어를 실시간으로 추출합니다. 이미지 복원 연산을 하지 않으므로 속도가 매우 빠릅니다.
- **LongSplat 내부의 Dual-Track 무시(Ignore) 메커니즘 탑재:**    
    - **Pose Stream:** 특징점 추적 모듈에서 $H_t > \tau$ 인 구역을 마스킹 아웃(Mask out)하여 오직 고정된 순수 배경으로만 카메라 궤적(Trajectory)을 풉니다.
    - **Photometric Stream:** 렌더링 손실 함수 계산 시 가중치 맵 $W_t = \exp(-\alpha H_t)$를 곱해 눈송이 영역을 무시합니다.
- **Spatial Neighboring 연산 모듈 추가:**
    - **구체화:** 구멍 난 눈송이 자리를 메우기 위해, 인접한 Clean 픽셀들의 가우시안 전파 가중치를 높이는 크로스 바이래터럴(Cross-Bilateral) 구조를 손실 함수 뒷단에 커스텀 레이어로 붙입니다.
