# Mio：适用于 Jetson AGX Orin 的实用化推理部署

Mio是多个LLM、VLM和TTS模型的缝合怪，各自的依赖相互冲突，因此需要做一点小小的魔改。本人主要在 Jetson AGX Orin 上开发并部署Mio，因此此处记载的信息仅供个人备忘。

```
# 克隆本仓库
git clone https://github.com/bd4sur/Nano
cd Nano

# 首先创建虚拟环境
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev
conda create -n nano python=3.10.15 pysocks -y
conda activate nano

# 单独安装NV官方魔改版PyTorch (https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
python -m pip install /home/bd4sur/software/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
python -m pip install /home/bd4sur/software/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
python -m pip install /home/bd4sur/software/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

# 编译安装llama-cpp-python
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" pip install . --verbose --force-reinstall --no-cache-dir

# 可选：安装 Flash-Attention-2
cd ..
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
# 接下来将`setup.py`中所有涉及`sm_xx`的部分改成`sm_87`（因为Orin的 Ampere GPU 的 Compute Capability == 8.7）
# 然后执行安装命令（并行任务数不能高于6，否则会耗尽内存。）
MAX_JOBS=4 python -m pip install . --no-build-isolation --verbose

# 安装Triton和AutoAWQ
cd ..
git clone https://github.com/triton-lang/triton
cd triton
python -m pip install ninja cmake wheel pybind11 # build-time dependencies
python -m pip install -e python --verbose
cd ..
git clone https://github.com/casper-hansen/AutoAWQ
cd AutoAWQ
python -m pip install . --verbose

# 安装其他依赖
python -m pip install -r requirements.txt --verbose
```

<details>

<summary>生成自签名SSL证书</summary>

**<span style="color: red;">警告：本节涉及网络安全，仅供本人技术备忘之用。读者切勿参考，否则后果自负。</span>**

由于现代浏览器的安全策略限制，必须使用HTTPS，才能在浏览器上使用语音交互。因此，在内网服务器上部署时，需要正确配置SSL证书。

1、首先生成私钥。过程中需要输入口令，必须牢记并保密该口令。

```
openssl genrsa -des3 -out key.pem 1024
```

2、在信任的环境中，将其解密为明文密钥，这样每次启动服务器或者建立SSL连接时，无需输入口令。

```
openssl rsa -in key.pem -out key_unencrypted.pem
```

3、生成CSR（证书签名请求）文件。注意：Common Name 必须与域名保持一致，否则浏览器会提示安全风险。

openssl req -new -key key_unencrypted.pem -out bd4sur.csr

4、生成自签名证书。首先在当前工作目录创建扩展配置文件`extconfig.txt`（[参考](https://www.openssl.org/docs/man3.0/man5/x509v3_config.html)），其内容如下，以添加“证书使用者可选名称”字段。如果不添加这一字段，则浏览器会提示安全风险。

```
basicConstraints = CA:FALSE
subjectAltName = @alt_names
[alt_names]
DNS.1 = ai.bd4sur.intra
```

然后执行以下命令，生成自签名证书`bd4sur.crt`，其有效期为365天。

```
openssl x509 -req -days 365 -in bd4sur.csr -signkey key_unencrypted.pem -out bd4sur.crt -extfile extconfig.txt
```

5、将私钥文件`key_unencrypted.pem`和证书文件`bd4sur.crt`置于以下目录：

- `~`
- `~/ai/funasr/models` 用于FunASR容器通过挂载的目录访问。

6、将证书设置为客户端的信任证书。对于安卓（小米）手机，通过“设置→密码与安全→系统安全→加密与凭据→从存储设备安装”，选取上面生成的`bd4sur.crt`，验证身份后，“凭据用途”选择“VPN和应用”即可。

7、在手机上设置域名解析。在内网搭建DNS服务器之前，以下是一个权宜手段：手机安装[Virtual-Hosts](https://github.com/x-falcon/Virtual-Hosts)，设置hosts文件并启动。

</details>

<details>

<summary>启动FunASR容器</summary>

```
# 首先启动容器
sudo docker run -p 10096:10095 -it --rm --privileged=true --name funasr \
--volume /home/bd4sur/ai/_model/FunASR:/workspace/models \
--workdir /workspace/FunASR/runtime \
registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12

# 在容器内启动FunASR进程
nohup /bin/bash run_server_2pass.sh \
--download-model-dir /workspace/models \
--vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
--model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx \
--online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
--punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
--itn-dir thuduj12/fst_itn_zh \
--hotword /workspace/models/hotwords.txt \
--certfile /workspace/models/bd4sur.crt \
--keyfile /workspace/models/key_unencrypted.pem > log.txt 2>&1 &
```

</details>

启动LLM服务器：`python server.py`

在浏览器中输入`https://ai.bd4sur.intra:8443`，进入对话窗口。

**注意事项**

- 视觉问答目前只支持针对一幅图片的连续问答。
- 选用纯语言模型时，不要上传图片，否则可能会出错。
