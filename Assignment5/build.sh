#!/bin/bash

if [ ! -d "examples" ]; then
    echo "you need data: examples.zip"
    exit 1
fi

if [ ! -d "build" ]; then
    mkdir "build"
fi
cd build
cmake ..
make -j8
cd ..