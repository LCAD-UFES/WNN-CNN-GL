#!/bin/bash

gnuplot --persist -e "plot '${1}' using 1:2 with linespoints"

#read -p "Press Enter to continue" </dev/tty
