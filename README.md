# M-LSD-onnxrun-cpp-py
使用ONNXRuntime部署面向轻量实时的M-LSD直线检测，包含C++和Python两个版本的程序

本套程序里在weights文件夹里有4个onnx模型文件，每个模型文件的大小不超过10M。
经过运行程序实验比较，model_512x512_large.onnx的检测效果最好。

本套程序对应的paper是顶会AAAI2022里的一篇文章《Towards Light-weight and Real-time Line Segment Detection》
