#!/bin/bash

source ../env/bin/activate
python generate_all_results.py | tee results.txt
