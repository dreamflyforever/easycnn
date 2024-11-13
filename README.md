### EasyCNN

This project aims to build a simple convolutional neural network optimized for limited CPU resources.

### Features

EasyCNN has a 3-layer deep convolutional structure, one pooling layer, and a sigmoid output for binary  
classification. The project includes two main parts: Forward Propagation and Backward Propagation.

### Training

To train the model, use the following command:

easycnn -t picture_folder

### Inference

To run inference on a single image, use:

easycnn -i picture

### flow chart
```mermaid
graph TD;
    A[训练] --> B[读取文件夹图片];
    B --> C[数据增强];
    C --> D[灰度填充640*640];
    D --> E[归一化处理];
    E --> F[3*3卷积];
    F --> G[池化];
    G --> H[3*3卷积];
    H --> I[池化];
    I --> J[1*1卷积];
    J --> K[ReLU激活函数];
    K --> L[更新参数(Adam优化器)];
    L --> M[与标注数据结果做均方误差];
    M --> N[判断是否收敛];
    N --> |未收敛| L;
    N --> |收敛| O[sigmoid或softmax输出];
```

### License

MIT License by Jim
