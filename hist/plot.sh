#!/bin/bash

for i in netflix; do
  for j in slice fiber; do
    for k in 0 1; do
      python ./hist.py ${i}_${j}${k}.log ${i}_${j}${k}.pdf
    done
  done
done
