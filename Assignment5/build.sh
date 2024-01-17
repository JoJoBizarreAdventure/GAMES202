#!/bin/bash

if [ ! -d "examples" ]; then
    echo "you need data: examples.zip"
fi

if [ ! -d "build" ]; then
    mkdir "build"
fi
cd build
cmake ..
make -j8
cd ..