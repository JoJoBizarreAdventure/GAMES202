#!/bin/bash

if [ ! -d "build" ]; then
    mkdir "build"
fi
cd build
cmake ../nori
make -j 16