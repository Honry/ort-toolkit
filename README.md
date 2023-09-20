# ort-toolkit
## Download a model
* on Linux
* set https_proxy to proxy-us
* pip3 install onnx onnxruntime
* git clone https://github.com/xenova/transformers.js/
* download a model via for example "python -m scripts.convert --model_id apple/mobilevit-small"

## Run WebNN
* Comment out "SHELL:-s MAXIMUM_MEMORY=4294967296" in onnxruntime/cmake/onnxruntime_webassembly.cmake
* Run browser with "--enable-experimental-web-platform-features --enable-blink-test-features --enable-features=WebnnMojoContext --enable-machine-learning-neural-network-api"
* Gaps. Browser implementation: computeSync and buildSync, more ops