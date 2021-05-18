net = importONNXNetwork('trail.onnx', 'OutputLayerType', 'classification');
I = imread("plastic93.jpg");
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));
i = classify(net,I);
label = ["cardboard","glass","metal","paper","plastic","trash"];
disp(label(i))