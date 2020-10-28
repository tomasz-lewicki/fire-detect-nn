FROM nvcr.io/nvidia/tensorrt:20.09-py3
RUN pip3 install torch torchvision requests
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
RUN cd torch2trt && python3 setup.py install


