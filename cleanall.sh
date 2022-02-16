#!/bin/bash

PROGLOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CURLOC=`pwd`

cd $PROGLOC

rm -rf otto/evaluate/outputs
rm -rf otto/learn/outputs
rm -rf otto/visualize/outputs
rm -rf otto/evaluate/tmp
rm -rf otto/learn/models

rm -rf __pycache__
rm -rf */__pycache__
rm -rf */*/__pycache__
rm -rf */*/*/__pycache__
rm -rf tests/.pytest_cache
rm -rf .idea
rm -rf */.idea
rm -rf */*/.idea
rm -rf */*/*/.idea

cd $CURLOC
