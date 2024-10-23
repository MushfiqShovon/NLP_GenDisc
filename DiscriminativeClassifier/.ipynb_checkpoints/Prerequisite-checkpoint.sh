echo "======================"
echo "Torchtext Installation"
echo "======================"
pip install torchtext==0.6.0
echo "======================"
echo "download en_core_web_sm"
echo "======================"
python -m spacy download en_core_web_sm
echo "======================"
echo "Installing Brevitas"
echo "======================"
pip install brevitas
echo "======================"
echo "Installing Pytorch-quantization"
echo "======================"
#pip install --no-cache-dir --index-url https://pypi.nvidia.com --index-url https://pypi.org/simple pytorch-quantization==2.1.3

# pip install onnx
# pip install onnx onnxruntime
# echo "======================"
# echo "Installing TensorRT"
# echo "======================"
# sudo apt-get update
# sudo apt-get install tensorrt
# sudo apt-get install python3-libnvinfer-dev
# export PATH=$PATH:/usr/src/tensorrt/bin/

#it it doesnt work, then sudo nano /etc/apt/sources.list.d/nvidia-ml.list
#Write 
       #deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /
#save the file

#Then in bash: sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
#sudo apt-get update
#sudo apt-get install tensorrt
#sudo apt-get install python3-libnvinfer-dev