#!/bin/bash

copyPrecompute(){
    echo "copy $1"
    if [ -f nori/scenes/cubemap/$1/light.txt ];then
        cp nori/scenes/cubemap/$1/light.txt homework2/assets/cubemap/$1
    fi
    if [ -f nori/scenes/cubemap/$1/transport.txt ];then
        cp nori/scenes/cubemap/$1/transport.txt homework2/assets/cubemap/$1
    fi
}

copyPrecompute CornellBox
copyPrecompute GraceCathedral
copyPrecompute Indoor
copyPrecompute Skybox