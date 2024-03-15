#! /bin/bash

curl 127.0.0.1:8080/generate \
  -X POST \
  -d '{"inputs":"What is deep learning?","parameters":{"max_new_tokens":2048,"temperature":0.7,"repetition_penalty":1}}' \
  -H 'Content-Type: application/json'
