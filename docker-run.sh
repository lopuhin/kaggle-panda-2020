docker run \
    --rm -it \
    --shm-size 8G \
    --security-opt=seccomp=unconfined \
    --gpus all \
     -v `pwd`/panda:/panda \
     -v `pwd`/data:/data \
     -v `pwd`/_runs:/panda/_runs/ \
     -v ~/.cache/torch:/root/.cache/torch/ \
     panda "$@"
