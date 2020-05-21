FROM lopuhin/panda-base

COPY . .

RUN pip3 install -e .
