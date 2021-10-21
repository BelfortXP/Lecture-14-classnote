##### tags: `COMP3340 Applied DL`

Table of contents:
- Introduction
- Benchmark
    - DAWNBench
    - MLPerf
- High level ideas and Representative methods
    - DL model complexity 
    - Roofline model
    - DL model Quantization
    - Operations behind DL model and design specific architecture in HLS
- Quiz
    - Analyze the model complexity of Lenet5 and apply it into Roofline model

# Lecture 14: Efficient Processing of DL models
In this section, an overview of the DL models's efficient processing will be introduced, inclusive of the background information, Benchmarks and some High level ideas and Representative methods. Two quizes will also be provided to help you test your understanding of the note.

## Benchmark
In this section, two benchmarks about deep learning training and inference will be introduced.

### DAWNBench
DAWNBench is a benchmark suite for end-to-end deep learning training and inference. Computation time and cost are critical resources in building deep models, yet many existing benchmarks focus solely on model accuracy. DAWNBench provides a reference set of common deep learning workloads for quantifying training time, training cost, inference latency, and inference cost across different optimization strategies, model architectures, software frameworks, clouds, and hardware.

<div align="center">  
<img src="https://dawn.cs.stanford.edu/assets/dawn-logo.svg" style="zoom:80%">
</div>

DAWNBench is also supported by many founding menbers like:

<div align="center">  
<img src="https://dawn.cs.stanford.edu/assets/logos/members/ant.png" style="zoom:20%">
<img src="https://dawn.cs.stanford.edu/assets/logos/members/facebook.png" style="zoom:60%">
<img src="https://dawn.cs.stanford.edu/assets/logos/members/google.png" style="zoom:20%">
<img src="https://dawn.cs.stanford.edu/assets/logos/members/vmware.jpg" style="zoom:30%">
</div>

### MLPerf
MLPerf is a consortium of AI leaders from academia, research labs, and industry whose mission is to “build fair and useful benchmarks” that provide unbiased evaluations of training and inference performance for hardware, software, and services—all conducted under prescribed conditions. To stay on the cutting edge of industry trends, MLPerf continues to evolve, holding new tests at regular intervals and adding new workloads that represent the state of the art in AI.

<div align="center">  
<img src="https://opendatascience.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-25-at-10.14.45-AM-640x300.png" style="zoom:80%">
</div>

MLPerf contains the benchmarks of both of the cloud and edge devices' training and inference. You can also check the latest benchmarks of MLPerf in this website [ML Commons](https://mlcommons.org/en/).

<div align="center">
<img src="https://raw.githubusercontent.com/BelfortXP/Lecture-14-classnote/main/MLPerf's%20content.png" style="zoom:60%">
</div>

One truthworthy point of MLPerf's benchmarks is it has a larget number of founding members, members and discussion groups (450+). Some of its foundering members are shown in there:

<div align="center">  
<img src="https://raw.githubusercontent.com/BelfortXP/Lecture-14-classnote/main/MLPerf's%20founding%20members.png">
</div>

## High level ideas and Representative methods
In this section, three kinds of high level ideas and their corresponding representative methods will be introduced.

### DL model complexity
The complexity of an algorithm are mainly focused on its time complexity and spatial complexity. For DL models, we use Floting-Point OPerations (FLOPs) to evaluate their time complexity. 

As the layers like pooling and activation contribute nearly zero FLOPs in many DL models, the FLOPs of a DL model are mainly contributed by Conv layer and Fully connected layer.

An example of convolution processing in Python in shown in there:
```python
def conv2d(img, kernel):
    height, width, in_channels = img.shape
    kernel_height, kernel_width, in_channels, out_channels = kernel.shape
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1
    feature_maps = np.zeros(shape=(out_height, out_width, out_channels))
    for oc in range(out_channels):              # Iterate out_channels (# of kernels)
        for h in range(out_height):             # Iterate out_height
            for w in range(out_width):          # Iterate out_width
                for ic in range(in_channels):   # Iterate in_channels
                    patch = img[h: h + kernel_height, w: w + kernel_width, ic]
                    feature_maps[h, w, oc] += np.sum(patch * kernel[:, :, ic, oc])
```

The visual representation of convolution processing can refer to the following figure:

<div align="center">  
<img src="https://miro.medium.com/max/990/1*DTTpGlhwkctlv9CYannVsw.gif" style="zoom:60%">
</div>

Through the code above, we can find that the FLOPs and memory used by convolution layer can be calculated by the functions below:

$$FLOPs = (kernel_{height}*kernel_{weight}*in_{channels}) * (out_{height}*out_{width}*out_{channels})$$
$$Memory\ cost = (kernel_{height}*kernel_{weight}*in_{channels}*out_{channels}) + (out_{height}*out_{width}*out_{channels})$$

Then an example of fully connected processing in Python in shown in there:
```python
def linear(in_data, kernel):
    in_size, out_size = kernel.shape
    feature_maps = np.zeros(shape=(out_size))
    for out_s in range(out_size):              
        feature_maps[out_s] = np.sum(in_data * kernel[:, out_s])
```

The visual representation of fully connected processing can refer to the following figure:

<div align="center">  
<img src="https://cs231n.github.io/assets/nn1/neural_net2.jpeg" style="zoom:50%">
</div>

Through the code above, we can find that the FLOPs and memory used by fully connected layer can be calculated by the functions below:

$$FLOPs = in_{size} * out_{size}$$
$$Memory\ cost = in_{size} * out_{size} + out_{size}$$

Appling the analysis above into the examples of VGG16 and MobileNet, the total FLOPs and memory cost of VGG16 and MobileNet will be like this:

<div align="center">
<b><font size=3>VGG16</font></b>
</div>
<div align="center">
<img src="https://pic3.zhimg.com/80/v2-d5753b4a2a91b3790165ff967cbc7fc2_1440w.jpg" style="zoom:50%">
</div>

<div align="center">
<b><font size=3>MobileNet</font></b>
</div>
<div align="center">
<img src="https://pic1.zhimg.com/80/v2-c0e8b7651af382009fadcd9a4a9a328c_1440w.jpg" style="zoom:45%">
</div>

### Roofline model
The roofline model is an performance model used to provide performance estimation of a given algorithm running on a specific hardware platform such as CPU, GPU and DNN acclerator.

There are two main part in roofline model, the algorithm we expected to analyze and the hardware platform we want to run the algorithm.

For the hardware platform, we will analyze its maximum attainable performance and maximum memory bandwidth. Then we can draw a figure like this one:

<div align="center">
<img src="https://pic2.zhimg.com/80/v2-cafb93b9a31fca2d7c84951555762e59_1440w.jpg" style="zoom:50%">
</div>

For the algorithm, we need to analyze its time complexity and space complexity and then compute its operational intensity. If the algorithm we want to analyze is a DNN model, then just like the FLOPs and model memory we analyzed in DL model complexity part. Then we can get the operational intensity <I>I</I> of the algorithm through this equation:

$$I = FLOPs_{in\ total} / Memory_{in\ total}$$

Take the VGG16 and MobileNet we analyzed in DL model complexity part, there operational intensity will be:

$$VGG16:\ I = 15,470,264,320 / 4*(138,334,128 + 15,262,696) \approx 25$$
$$MobileNet:\ I = 568,740,352 / 4*(4,209,088 + 15,284,664) \approx 7$$

Then for the NVIDIA GeForce GTX 1080 Ti which has a maximum attainable performance of 11.3TFLOP/s and a maximum memory bandwidth of 484GB/s, we can plot a figure of roofline model like this:

<div align="center">
<img src="https://pic3.zhimg.com/80/v2-55052a705e6225321fb562cd7d04283a_1440w.jpg" style="zoom:50%">
</div>

From the figures above, we can find that there are two kinds of bound of the algorithm's performance which are memory bound and compute bound.

Take the DL model as example again, there are some thoughts to handle the memory bound and compute bound.

<I>Memory Bound:</I>
- Increase the operational intensity of model
- Utilize data reuse to optimize computation process
- Design the specific hardware architecture
- Do the in-memory computing

<I>Compute Bound:</I>
- Decrease the operational intensity of model
- Increase the computational power of hardware
- Design the specific hardware architecture
- Design the specific computation unit

## Quiz
### 1. Analyze the model complexity of Lenet5 and apply it into Roofline model
<b>1.1 Analyze the Kernel Memory, Output Memory and FLOPs of each layer of Lenet5 (Don’t need to consider bias)</b>

<div align="center">
<img src="https://miro.medium.com/max/2597/1*y68ztClLF6ae7P53ayyFzQ.png" style="zoom:50%">
</div>

Analysis of Lenet5:

| Layer Type | Kernel Size | Kernel Mem | Output Size | Output Mem | FLOPs |
| ---------- | ----------- | ---------- | ----------- | ---------- | ------|
| InputLayer | 0 | 0 | (32, 32, 1) | 1024 | 0 |
| Conv2D | (5, 5, 1, 6) | 150 | (28, 28, 6) | 4706 | 117600 |
| Subsampling | 0 | 0 | (14, 14, 6) | 1176 | 0 |
| Conv2D | (5, 5, 6, 16) | 2400 | (10, 10, 16) | 1600 | 240000 |
| Subsampling | 0 | 0 | (5, 5, 16) | 400 | 0 |
| Conv2D | (5, 5, 16, 120) | 48000 | 120 | 120 | 48000 |
| Dense | (120, 84) | 10080 | 84 | 84 | 10080 |
| Dense | (84, 10) | 840 | 10 | 10 | 840 |
| --- | --- | --- | ---| --- | --- |
| Summary | --- | 61470 | --- | 9120 | 416520 |

<b>1.2 For Lenet5 model running in GTX 1080 Ti, please calculate its operational intensity and peak performance</b>

For GeForce GTX 1080 Ti graphics card, we have:
- Maximum attainable performance = 11.3 TFLOP/s
- Maximum memory bandwidth = 484 GB/s

For Lenet5, we have:
- Operational intensity of Lenet5 = 416520 FLOPs / 4*(61470+9120) Byte ≈ 1.475 FLOPs/Byte
- Peak performance in 1080 Ti will be 1.475*484 = 0.714 TFLOP/s

<b>1.3 Please specify the bound (memory or compute) of the process in 1.2 and provide some thoughts to improve it</b>

It is in memory bound.

Thoughts:
- Increase the operational intensity of model
- Utilize data reuse to optimize computation process
- Design the specific hardware architecture
- Do the in-memory computing


