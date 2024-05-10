#!/bin/bash

if [ "$1" = "gpu" ]
then
    partition="disa"
elif [ "$1" = "cpu" ]
then
    partition="cpu2019-bf05"
else
    echo "Invalid argument. Usage: $0 gpu|cpu [partition]"
    exit 1
fi

selected_partition=${2:-$partition}

if [ "$1" = "gpu" ]
then
    echo "Allocating GPU partition $selected_partition..."
    salloc --mem=50G -t 04:59:00 -p "$selected_partition" --gres=gpu:1
else
    echo "Allocating CPU partition $selected_partition..."
    salloc --mem=50G -c 1 -N 1 -n 1 -t 04:00:00 -p "$selected_partition"
fi