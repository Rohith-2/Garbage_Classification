videoFileReader = VideoReader('/Users/rohith/Downloads/IMG_9236.MOV');
myVideo = VideoWriter('myFile_1.avi');
net = importONNXNetwork('trail.onnx', 'OutputLayerType', 'classification');
label = ["cardboard","glass","metal","paper","plastic","trash"];


% Setup: create deployable video player and face detector
depVideoPlayer = vision.DeployableVideoPlayer;
open(myVideo);

while hasFrame(videoFileReader)

	% read video frame
	I = readFrame(videoFileReader);
	% process frame
%I = imread(videoFrame);
inputSize = net.Layers(1).InputSize;
II = imresize(I,inputSize(1:2));
i = classify(net,II);
position = [650 400];
RGB = insertText(I,position,label(i),'FontSize',18,'BoxColor',...
    'red','BoxOpacity',0.4,'TextColor','white');
	% Display video frame to screen
	%depVideoPlayer(videoFrame);

	% Write frame to final video file
	writeVideo(myVideo,RGB);
	pause(1/videoFileReader.FrameRate);

end
close(myVideo)