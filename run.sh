#!/bin/sh

#python3 run.py --lr 4e-5

#python3 run.py --lr 4e-5

#python3 run.py --lr 4e-5

#python3 run.py --lr 1e-6


trap "kill 0" 2

#count=`ps aux | grep python | grep run.py | grep -v grep | wc -l`
#while [ "$count" = "2" ]
#do
#  count=`ps aux | grep python | grep run.py | grep -v grep | wc -l`
#  echo "zzz"
#  sleep 1800
#done

#echo "happy to start!"
python3 run.py



