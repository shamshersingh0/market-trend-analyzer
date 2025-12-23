#!/bin/bash
# setup.sh
apt-get update && apt-get install -y libgomp1
pip install --upgrade pip setuptools wheel