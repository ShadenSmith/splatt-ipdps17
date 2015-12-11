#!/bin/bash

NAME=$1

../splatt cpd --nowrite -v -t 36 ~/tensor/${NAME}.csf > ${NAME}.csf.log

sort -n slice0.hist | uniq -c > ${NAME}_slice0.hist
sort -n slice1.hist | uniq -c > ${NAME}_slice1.hist
sort -n fiber0.hist | uniq -c > ${NAME}_fiber0.hist
sort -n fiber1.hist | uniq -c > ${NAME}_fiber1.hist
mv slice0.hist ${NAME}_slice0.log
mv slice1.hist ${NAME}_slice1.log
mv fiber0.hist ${NAME}_fiber0.log
mv fiber1.hist ${NAME}_fiber1.log
