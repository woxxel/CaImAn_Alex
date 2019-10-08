#!/usr/bin/env python

import argparse
import distutils.dir_util
import filecmp
import os
import shutil
import string
import subprocess
import sys # for sys.prefix

from caiman.paths import caiman_datadir

sourcedir_base = os.path.join(sys.prefix, "share", "caiman") # Setuptools will drop our datadir off here

###############
# caimanmanager - A tool to manage the caiman install
#
# The caiman data directory is a directory, usually under the user's home directory
# but configurable with the CAIMAN_DATA environment variable, that is used to hold:
#   - sample movie data
#   - code samples
#   - misc json files used by Caiman libraries
#
# Usually you'll want to work out of that directory. If you keep upgrading Caiman, you'll
# need to deal with API and demo changes; this tool aims to make that easier to manage.

##########
# For in-place caiman installs (which will likely only be used by developers), we need
# to come up with a similar subset of files that setup.py would install for a normal pip install,
# focused around the data directory.
extra_files = ['test_demos.sh', 'README.md', 'LICENSE.txt']
extra_dirs  = ['demos', 'docs', 'model', 'testdata']
standard_movies = [
		os.path.join('example_movies', 'data_endoscope.tif'),
		os.path.join('example_movies', 'demoMovie.tif')]

###############
# commands

def do_install_to(targdir, inplace=False, force=False):
	global sourcedir_base
	if os.path.isdir(targdir) and not force:
		raise Exception(targdir + " already exists")
	if not inplace: # In this case we rely on what setup.py put in the share directory for the module
		if not force:
			shutil.copytree(sourcedir_base, targdir)
		else:
			distutils.dir_util.copy_tree(sourcedir_base, targdir)
	else: # here we recreate the other logical path here. Maintenance concern: Keep these reasonably in sync with what's in setup.py
		for copydir in extra_dirs:
			if not force:
				shutil.copytree(copydir, os.path.join(targdir, copydir))
			else:
				distutils.dir_util.copy_tree(copydir, os.path.join(targdir, copydir))
		os.makedirs(os.path.join(targdir, 'example_movies'), exist_ok=True)
		for stdmovie in standard_movies:
			shutil.copy(stdmovie, os.path.join(targdir, 'example_movies'))
		for extrafile in extra_files:
			shutil.copy(extrafile, targdir)
	print("Installed " + targdir)

def do_check_install(targdir, inplace=False):
	global sourcedir_base
	if inplace:
		sourcedir_base = os.getcwd()
	ok = True
	comparitor = filecmp.dircmp(sourcedir_base, targdir)
	alldiffs = comparitor_all_diff_files(comparitor, '.')
	if alldiffs != []:
		print("These files differ: " + " ,".join(alldiffs))
		ok = False
	leftonly = comparitor_all_left_only_files(comparitor, ".")
	leftonly = list(map(lambda fn: os.path.normpath(fn), leftonly))
	if inplace: # Need to filter down list
		leftonly = list(filter(lambda fn: (fn in extra_files) or					# if it is an explicit extra_files in the root of the sourcetree
					(fn in list(map(lambda mov: os.path.join(mov), standard_movies))) or	# if it is one of the standard movies
					(fn.split(os.sep)[0] in extra_dirs), leftonly ))			# if it is ANY file in one of the predefined directories
	if leftonly != []:
		print("These files don't exist in the target:\n\t" + "\n\t".join(leftonly))
		ok = False
	if ok:
		print("OK")

def do_run_nosetests(targdir):
	out, err, ret = runcmd(["nosetests", "--traverse-namespace", "caiman"])
	if ret != 0:
		print("Nosetests failed with return code " + str(ret))
		sys.exit(ret)
	else:
		print("Nosetests success!")

def do_run_demotests(targdir):
	out, err, ret = runcmd([os.path.join(caiman_datadir(), "test_demos.sh")])
	if ret != 0:
		print("Demos failed with return code " + str(ret))
		sys.exit(ret)
	else:
		print("Demos success!")

def do_nt_run_demotests(targdir):
	# Windows platform can't run shell scripts, and doing it in batch files
	# is a terrible idea. So we'll do a minimal implementation of run_demos for
	# windows inline here.
	os.environ['MPLCONFIG'] = 'ps' # Not sure this does anything on windows
	demos = glob.glob('demos/general/*.py') # Should still work on windows I think
	for demo in demos:
		print("========================================")
		print("Testing " + str(demo))
		if "demo_behavior.py" in demo:
			print("  Skipping tests on " + demo + ": This is interactive")
		else:
			out, err, ret = runcmd(["python", demo], ignore_error=False)
			if ret != 0:
				print("  Tests failed with returncode " + str(ret))
				print("  Failed test is " + str(demo))
				sys.exit(2)
			print("===================================")
	print("Demos succeeded!")

###############
#

def comparitor_all_diff_files(comparitor, path_prepend):
	ret = list(map(lambda x: os.path.join(path_prepend, x), comparitor.diff_files)) # Initial
	for dirname in comparitor.subdirs.keys():
		to_append = comparitor_all_diff_files(comparitor.subdirs[dirname], os.path.join(path_prepend, dirname))
		if to_append != []:
			ret.append(*to_append)
	return ret

def comparitor_all_left_only_files(comparitor, path_prepend):
	ret = list(map(lambda x: os.path.join(path_prepend, x), comparitor.left_only)) # Initial
	for dirname in comparitor.subdirs.keys():
		to_append = comparitor_all_left_only_files(comparitor.subdirs[dirname], os.path.join(path_prepend, dirname))
		if to_append != []:
			ret.append(*to_append)
	return ret

###############

def runcmd(cmdlist, ignore_error=False, verbose=True):
	# In most of my codebases, runcmd saves and returns the output.
	# Here I've modified it to send right to stdout, because nothing
	# uses the output and because the demos sometimes have issues
	# with hanging forever
        if verbose:
                print("runcmd[" + " ".join(cmdlist) + "]")
        pipeline = subprocess.Popen(cmdlist, stdout = sys.stdout, stderr = sys.stdout)
        (stdout, stderr) = pipeline.communicate()
        ret = pipeline.returncode
        if ret != 0 and not ignore_error:
                print("Error in runcmd")
                print("STDOUT: " + str(stdout))
                print("STDERR: " + str(stderr))
                sys.exit(1)
        return stdout, stderr, ret

###############
def main():
	cfg = handle_args()
	if   cfg.command == 'install':
		do_install_to(cfg.userdir, cfg.inplace, cfg.force)
	elif cfg.command == 'check':
		do_check_install(cfg.userdir, cfg.inplace)
	elif cfg.command == 'test':
		do_run_nosetests(cfg.userdir)
	elif cfg.command == 'demotest':
		if os.name == 'nt':
			do_nt_run_demotests(cfg.userdir)
		else:
			do_run_demotests(cfg.userdir)
	elif cfg.command == 'help':
		print("The following are valid subcommands: install, check, test, demotest")
	else:
		raise Exception("Unknown command")

def handle_args():
	global sourcedir_base
	parser = argparse.ArgumentParser(description="Tool to manage Caiman data directory")
	parser.add_argument("command", help="Subcommand to run. install/check/test/demotest")
	parser.add_argument("--inplace", action='store_true', help="Use only if you did an inplace install of caiman rather than a pure one")
	parser.add_argument("--force", action='store_true', help="In installs, overwrite parts of an old caiman dir that changed upstream")
	cfg = parser.parse_args()
	if cfg.inplace:
		# In this configuration, the user did a "pip install -e ." and so the share directory was not made.
		# We assume the user is running caimanmanager right out of the source tree, and still want to try to
		# copy the correct files out, which is a little tricky because we never kept track of that before.
		sourcedir_base = os.getcwd()
	cfg.userdir = caiman_datadir()
	return cfg

###############

main()
