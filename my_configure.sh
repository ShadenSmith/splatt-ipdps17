#!/bin/bash

pushd src/SpMP; make -j ; popd
./configure --cc=icc --cxx=icpc $@ --dev
