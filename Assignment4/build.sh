#!/bin/bash

cd lut-gen

if [ ! -d "build" ]; then
    mkdir "build"
fi
cd build
cmake ..
make -j 16