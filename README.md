## Description
Solving Face detection & Facial landmark detection through mtcnn(Multitask cascade convolutional neural network).
## Testing and predict
[Path Setting]
	export PYTHONPATH=$PYTHONPATH: <your path>/mtcnn_tensorflow/detection

[With only images]
1. Copy your image file to `testing/images`
2. Run `python testing/test_images.py --stage=onet`. Anyway you can specify stage to pnet or rnet to check your model.
3. The result will output in `testing/results_onet`

[With WebCam]
1. Connect your webcamera to the system.
2. Run `python testing/test_video.py --stage=onet` --profile=False.
    Enable Profiling by --profile=True.
    



