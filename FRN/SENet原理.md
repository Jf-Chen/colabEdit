对于一张输入图像X，大小是H' x W' x C'，经过C个卷积核，得到 H x W x C

$V=[v_1,,,,v_C]$ 代表卷积核集合，$U=[u_1,,,,u_C]$ 是输出，$u_C \in R^{H \times W}$ 是一个通道的特征，v_C表示单个通道

#### Squeeze

每个卷积核都有一个局部感受野，也就是每个u_C都不能利用对应区域外的上下文信息。

将HxWxC压缩成1x1xC的输出，表明C个feature map的数值分布情况，视为全局信息。

#### Excitation: Adaptive Recalibration

将1x1xC的z进行压缩reductio倍，得到1x1x(C/r)，经过ReLU激活，再还原成1x1xC的大小。这样做而不是用全连接的目的是

1）具有更多的非线性，可以更好地拟合通道间复杂的相关性；

2）极大地减少了参数量和计算量

然后1x1xC通过一个sigmod门获得归一化权重，最后乘到原来的特征上完成加权。