#!/bin/bash

##########################
# test_demos.sh
#
# This is intended to test the python (.py) demos
#
# dependencies:
#   xvfb-run - This starts a dummy X server for anything that might
#              need to plot things or show movies. Note that any demos
#              we test MUST NOT wait for input while showing a movie,
#              because in the test environment nobody will show up to provide
#              that input (like clicking things or hitting q)

# Make sure the xvfb-run command exists before starting demos, so we can give a better
# error message
OS=$(uname -s)


export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


# On Linux we wrap our command with xvfb-run to allow X things to happen
# and make them effectively no-ops. Not necessary on other platforms.

if [ $OS == "Linux" ]; then
	command -v xvfb-run
	if [ $? != 0 ]; then
		echo "xvfb-run command not found"
		exit 1
	fi
fi

# Tell matplotlib to try to plot less to begin with by specifying a postscript backend
export MPLCONFIG=ps

cd `dirname ${BASH_SOURCE[0]}`

for demo in demos/general/*; do
	if [ $demo == "demos/general/demo_behavior.py" ]; then
		echo "	Skipping tests on $demo: This is interactive"
	elif [ -d $demo ]; then
		true
	else
		echo Testing demo [$demo]
		echo Timestamp is $(date +%s)
		if [ $OS == "Linux" ]; then
			xvfb-run --server-args='-screen 0 1024x768x24' -a python $demo
		else
			python $demo
		fi
		err=$?
		if [ $err != 0 ]; then
			echo "	Tests failed with returncode $err"
			echo "	Failed test is $demo"
			echo Timestamp is $(date +%s)
			exit 2
		fi
		echo Timestamp is $(date +%s)
		echo "=========================================="
	fi
done
