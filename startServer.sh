#!/bin/bash

if [! $# == 1 ] 
then
    exit 1
fi

http-server homework$1 -p 8000 -c-1
