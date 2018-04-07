# 03.30 - SqueezeNet Implementation

[https://github.com/MachineLearningOfThings/mot-dev](https://github.com/MachineLearningOfThings/mot-dev)

# 앞서 알아야 할 기본

- Global Average Pooling
- Network In Network
  - 참고 - <[https://goo.gl/AGRHu4](https://goo.gl/AGRHu4)>

# 작은 CNN 구조가 필요한 이유

- **More efficient distributed training.** 적은 파라미터로 Model의 속도를 빠르게 만들고자 한다.
- **Less overhead when exporting new models to clients.** 서버에서 OTA(Over the air)를 통해 학습된 모델을 Update 하고자 할 때 모델의 용량을 줄이고자 한다.
- **Feasible FPGA and embedded deployment.** 보통 10MB 보다 적은 FPGA(재프로그래밍 가능한 실리콘 칩)에 넣을 수 있도록 하고자 한다.

적은 파라미터 → 빠른 속도와 적은 용량 

전략. 1 - 3x3 필터를 1x1 필터로 대체한다. 

- 3x3 보다는 파라미터가 9배가 적은 1x1를 사용하도록 한다.

전략. 2 - 3x3 Filter의 input channel 갯수를 줄여준다. 

- Fire Module에 Input으로 들어갈 때 Channel 수가 확줄어들면서 생기는 bottleneck을 방지 해주기 위해 1x1를 사용한다.
- 채널 갯수도 많은데 filter의 사이즈가 커진다면, 그 만큼 제곱으로 더 커질 것이다. 그렇기 때문에 앞에서 채널 갯수를 줄여주고 3x3 filter의 convolution layer에 넣어주도록 한다.

## Architecture

![](https://static.notion-static.com/731fa442-e834-472b-84f2-1140d7ba8f4b/Untitled)

위는 SqueezeNet 논문에서 제시한 3가지 구조다. 

가장 왼쪽은 가장 기본적인 구조이고, 가운데와 오른쪽은 좀 더 응용해서 Connection을 한 구조. 

보통 가운데와 오른쪽 사진처럼 화살표가 표시 되어있는 그림을 보면 대부분 ResNet과 같은
Connection 구조라 생각하면 된다. 
여기서는 Element Wise Add를 하려면 더하는 두 Layer의 Channel 갯수가 맞아야 하기 때문에 전체 모델에서 절반만 연결하는 구조를 짤 수 밖에 없었다고 한다. 

오른쪽 사진은 한번 더 더하기 위해 conv_1x1를 거쳐 channel 수를 맞추고서 더한다. 

그럼 여기서 더 할 때 Element Wise로 더하냐 또는 곱하는지에 따라 고민해볼 수 있는데... 

![](https://static.notion-static.com/da07c760-e2ee-4452-95de-78b1ae26812a/Untitled)

다행히 뒤에 친절하게 어떻게 했는지 간략히 설명 해주었다. 

![](https://static.notion-static.com/a917ce2d-5eae-4911-938b-2590372d33eb/Untitled)

위의 사진은 좀 더 디테일하게 구조 설명을 해준 것이다. 

사실상 SqueezeNet의 모든 구조를 여기서 설명 해주었다고 생각하면 된다. 
다만 여기서 자세히 적혀있지 않은 것이 있는데. 그것은 fire의 의미, 그리고 Activation Function에 대해서는 자세히 적혀 있지 않다. 
그래도 일단 저 Table로 전체적인 구조를 생각하면 될 것이다. 

Output Size는 SqueezeNet 논문에서 사용한 데이터 이미지 사이즈의 예시이니 이해하는 용도로 사용하면 되고, 오른쪽에 (96, 128, 256, 384 ... )를 보면 된다. 이것이 Filter 개수 (또는 Channel)를 말하는 것이며, 이것으로 Filter 갯수를 정해주면 되겠다. 
그리고 마지막 1000은 Class의 갯수라고 생각하면 된다. 

# Fire Module

![](https://static.notion-static.com/47fde3d3-64ea-42a1-b08c-f59584853c22/Untitled)

SqueezeNet에서 다른 모델과의 가장 큰 차이점은 **Squeeze Layer**와 **Expand Layer**가 있다는 것이다. 

ResNet이나 GoogLeNet이나 Network In Network 같은 유명한 모델들은 Convolution Filter 사이즈를 1x1과 3x3를 썼다는 것인데. 여기선 1x1와 3x3에서 얻은 Output을 합쳐 두 장점을 얻고, 깊이 쌓지 않음으로 Parameter의 갯수를 최소화 하였다. 

**Squeeze Layer**에서 먼저 전략.1 대로 3x3 보다는 parameter 가 9배 적은 1x1로 특징들을 뽑았다. 그리고 그 뒤에 ReLU를 거친다는 것이 그림에서 친절하게 나온다. Squeeze Layer는 Expand Layer에 들어가는 Input Layer의 Channel 갯수를 제한하는 역활을 한다. 

그 뒤로 **Expand Layer**를 거치는데. 여기서는 1x1 와 3x3 filter size가 나온다. 여기서는 1x1과 3x3 filter를 거친 convolution layer를 concatenate 한다. 다시 말해서 channel 끼리 붙여서 내보낸다는 것이다. 

![](https://static.notion-static.com/c2cec073-a22f-4817-890d-3b73bf452c66/Untitled)

# Detail Architecture

큰 모델 구조는 알았고, Convolution Block 하면 함께 따라오는 Activation Function, 또 다른 세세한 Parameter를 알아봐야겠다. 

![](https://static.notion-static.com/2be14778-3319-44b7-9811-7db2497f66d7/Untitled-1.jpg)

왠만하면 Padding 이야기가 없으면 각 Layer 마다 Output Size가 2-4씩 줄어들지 않는 이상 대부분 TensorFlow에서 SAME이라고 생각하면 되는 거고, 두번째에 ReLU를 각 Squeeze와 Expand Layer 뒤에 사용했다고 적혀 있다. 

위의 Table에서는 보여주어지지 않았지만, 그 아래에 Dropout도 Fire9 Module 뒤에 사용했다는 것을 참고 하면 되겠다. 

# Architecture

그럼 아래는 전체적인 Squeeze Net 구조다. 

![](https://static.notion-static.com/659e94c1-8965-44dd-8029-ede552ea182f/Untitled)

위에서 보여주었던 Table 대로 처음엔 convolution layer와 activation function (ReLU)를 넣어주었고, 
그 다음부터 Fire Module들이 들어간다. 각 Fire Module에는 Squeeze 와 Expand Layer들이 들어가고, 한 블럭이 지나갈 때 마다 Max Pooling을 해준다. 

각 Convolution과 Pooling Layer들의 Filter의 갯수와 사이즈만 주의해서 넣어주면 된다. Activation Function은 Convolution Layer 어느 뒤에 들어가는지 꼭 확인해서 넣어주어야 할 것이고(혹시 안 적혀있으면 Convolution 바로 뒤에 넣어주면 된다.), Pooling은 꼭 Convolution Layer 뒤에 나오는 것은 아니니 얼마만에 나오는지만 신경 써주면 될 것이다. (왠만한 논문에는 친절히 설명해준다)

# Metaparameters

![](https://static.notion-static.com/a28ed4a7-c751-4ae2-8c92-b862d65b2a04/Untitled)

모델 깊이에 따라 모델은 무거울 수 있지만 데이터에 따라 더 좋은 성능을 낼 수 있고, 

논문에서는 SqueezeNet 모델에 대해 다양한 Parameter들을 알려준다. 

- base_e = 128
- incr_e = 128
- pct_3x3 = 0.5
- freq = 2
- SR = 0.125

## Squeeze Ratio (SR)

# Compression & Result

![](https://static.notion-static.com/01732871-5d34-46dc-8f56-9e3c3918d76b/Untitled)

![](https://static.notion-static.com/e2a1842d-a101-41ba-bc3a-86bad53d7e52/Untitled)

위의 테이블은 AlexNet과 SqueezeNet을 Compression와 Accuracy의 결과다. 
여기서 보면 SVD, Network Prunning, Deep Compression을 사용 할 때 Deep Compression이 가장 Accuracy를 잃지 않으면서 가장 크게 줄였다고 한다. 

그리고 Deep Compression은 보통 모델 사이즈가 클 때 더 좋은 효과를 낸다고 하지만, SqueezeNet은 모델 사이즈가 작음에도 불구하고 높은 압축률을 보여주었다고 한다. 

# 앞으로 공부하면 좋을만한 것들

## Compression Methods

- Deep Compression - [https://arxiv.org/abs/1510.00149](https://arxiv.org/abs/1510.00149)
  - [https://fuzer.github.io/Compressing-and-regularizing-deep-neural-networks/](https://fuzer.github.io/Compressing-and-regularizing-deep-neural-networks/)
  - [http://www.navisphere.net/5481/deep-compression-compressing-deep-neural-networks-with-pruning-trained-quantization-and-huffman-coding/](http://www.navisphere.net/5481/deep-compression-compressing-deep-neural-networks-with-pruning-trained-quantization-and-huffman-coding/)
-