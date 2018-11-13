#!/bin/bash
echo "Killing all docker containers with a name  matching ${USER}_maddpg_GPU_*"
docker rm $(docker stop $(docker ps -a -q --filter name=${USER}_maddpg_GPU_ --format="{{.ID}}"))
