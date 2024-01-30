FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel  

MAINTAINER neov neov@neov.im

ENV APP_DIR=/workspace

RUN mkdir -p -m 777 $APP_DIR

RUN pip install --no-cache-dir \ 
transformers==4.28.1 numpy rouge_score fire openai sentencepiece tokenizers==0.13.3 wandb deepspeed==0.8.0 accelerate tensorboardX  \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
--trusted-host pypi.tuna.tsinghua.edu.cn 

COPY apex ${APP_DIR}/apex

RUN apt-get update && apt-get -y install nvidia-cuda-dev

RUN cd ${APP_DIR}/apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && cd .. &&  rm -rf ${APP_DIR}/apex

WORKDIR $APP_DIR