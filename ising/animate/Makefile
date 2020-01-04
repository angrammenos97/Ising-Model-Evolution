# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = g++

all: Ising_Animation

Ising_Animation: animation.cpp
	g++ -I /usr/include/opencv2/ -L /usr/lib/ -o $@ $< -lopencv_core -lopencv_highgui

clean:
	$(RM) Ising_Animation output*.bin *~
