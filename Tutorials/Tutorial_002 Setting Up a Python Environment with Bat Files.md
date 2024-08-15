This tutorial explains how to set up and use [ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat), a script that combines WinPython and ConEMU to create a customizable Python environment on Windows. Double clicking on the ConEMU.bat file will set up the environment so that the working directory is in the same location as the bat file.

This provides several advantages, such as better navigation, consistent scripting environment (e.g. specific version of Python and libraries) and possible security improvements. 

# Table of Contents
- [Introduction](#introduction)
  - [What is WinPython](#what-is-winpython)
  - [How WinPython Scripts work](#how-winpython-scripts-work)
  - [What techniques we took from WinPython](#what-techniques-we-took-from-winpython)
- [How to use ConEMU.bat](#how-to-use-conemubat)
  - [What is ConEMU?](#what-is-conemu)
  - [Prerequisites](#prerequisites)
  - [Outline of Steps to use ConEMU.bat](#outline-of-steps-to-use-conemubat)
    - [Step #1: Copy ConEMU.bat into the WinPython scripts directory](#step-1-copy-conemubat-into-the-winpython-scripts-directory)
    - [Step #2: Copy Desktop_Cmd.bat to the desired Python working directory](#step-2-copy-desktop_cmdbat-to-the-desired-python-working-directory)
- [Motivation for ConEMU.bat](#motivation-for-conemubat)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)


# Introduction

As mentioned in the previous tutorial, there are many ways to set up a shell environment with a python interpreter. For instance, if you install, python from python.org, the installation script will likely include the path to the python interpreter as an environmental variable. Linux package managers typically do something similar when installing python. You can use CygWin to have a Linux like shell environment within windows. 

One downside to this approach is that only one python interpreter can be mapped to the python command. Rather than having preconfigured environmental (either system wide or per user) variables for python we can run some code to set up a custom shell environment for python. Both WinPython and Anacanda use this approach. 

## What is WinPython

WinPython is a free open-source portable distribution of the Python programming language for Windows 7/8/10 and scientific and educational usage. It includes many scientific and data science-oriented packages [1]. WinPython is designed to be used without installation, allowing users to run Python applications on any Windows computer.

## How WinPython Scripts work

In the case of WinPython the environment is set up via customizable script files. The basic environment (e.g. the location of the python interpreter is found in the file):

`C:\WPy64-31050\scripts\env.bat

This file isn't called directly instead it is called via a wrapper script that does some other minor house keeping such as defining the working directory. This wrapper script is:

` C:\WPy64-31050\scripts\env_for_icons.bat

WinPython comes with several scripts to set up the python environment in various shell environments (e.g. cmd.exe and PowerShell) as well is in iron python notebooks and a qtconsole. 

## What techniques we took from WinPython

In this project we provided an optional .bat file (called [ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat)) to set up the python environment that borrows from the above mentioned scripts (e.g. cmd.bat) that come with WinPython. The format for the bat file in this project (i.e. [ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat)) follows the same format as the cmd.bat provided by WinPython, but the last line is changed from:

` cmd.exe /k
 
 to 

` C:\Program^ Files\ConEmu\ConEmu64.exe cmd

so that the ConEMU terminal is used instead of cmd.exe This terminal provides better readability and copy and paste features. Some information about ConEMU can be found at:
https://www.pearltrees.com/s243a/conemu-terminal-emulator/id71458782

# How to use ConEMU.bat

## What is ConEMU?

ConEMU is a Windows console emulator with tabs, which presents multiple consoles and simple GUI applications as one customizable GUI window with various features.

## Prerequisites (How to use ConEMU.bat)

Before using ConEMU.bat both WinPython and ConEMU should be installed. For installation instructions of ConEMU see:

https://conemu.github.io/en/Installation.html 

 WinPython can be downloaded and installed from:
 
 https://winpython.github.io/ 

One could likely apply the above mentioned WinPython scripts to other versions of python, if they didn't want to use WinPython. Unfortunately, these files aren't separate files in the github repo but instead generated via a make file. So for instance the code for cmd.bat is buried at:
https://github.com/winpython/winpython/blob/f7aed01035cec750b9580b9a79a1f20e880141cd/make.py#L1416

but if one installs WinPython they can use the build scripts for other python environments. 

## Outline of Steps to use ConEMU.bat

Once the above prerequisites are met we can use "[ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat)". The steps are as follows:

##### Step #1, copy  ConEMU.bat into the winpython script directory

The scripts directory is an immediate subfolder of the WinPython installation. So if WinPython was installed at "C:\WPy64-31050", then the path to the scripts directory would be:
` C:\WPy64-31050\scripts

ConEMU.bat must be placed in this location because this script expects env.bat to be in the same folder (see the advanced BAT script section). 

##### Step #2 Copy Desktop_Cmd.bat to the desired python working directory.

When a BAT file is opened the working directory is set to the location of the BAT file, so the puporse of [Desktop_Cmd.bat]([PT_Tools/PT_Tools/Desktop_Cmd.bat at main · s243a/PT_Tools (github.com)](https://github.com/s243a/PT_Tools/blob/main/PT_Tools/Desktop_Cmd.bat)) is simply to set the working directory. However, an advance user could modify this script to add any additional project specific commands to set up the shell environment. 

This script [Desktop_Cmd.bat]([PT_Tools/PT_Tools/Desktop_Cmd.bat at main · s243a/PT_Tools (github.com)](https://github.com/s243a/PT_Tools/blob/main/PT_Tools/Desktop_Cmd.bat)), uses the CALL command, so that a new process is not started when [ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat) is called. This keeps the working directory the same, but [ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat). [ConEMU.bat](https://github.com/s243a/PT_Tools/blob/main/WPy/ConEMU.bat) finds env.bat by the command, 

` call "%~dp0env_for_icons.bat"  %*

"%~dp0", is a special BAT file syntax, which means the path that follows is relative to the location of the bat file. The call command launches the instructions in the bat file within the same process rather than starting a new process.

# Motivation for ConEMU.bat

This script is motivated to make testing python scripts easy, with this .bat file you can put the python code your working on anyware (e.g. in a dropbox directory) and simply double click on the bat file to launch your testing environment. 

In addition to ease of testing, the script provides a way of easily configuring python to the test environment. For instance advanced users could use the pip-env command to have a specific version of python specified as well as specific version of any requisite python packages. 

Finally, there are possible security advantages to setting up the python environment using a script then specifying it as an environmental variable. For instance it may make it harder for hackers to discover that python is installed especially if some of these files are in directories that are unsearchable to the user. For security reasons android makes some directories unsearchable. 

# Troubleshooting

- If ConEMU doesn't launch, ensure it's correctly installed and the path in ConEMU.bat is correct.
- If Python isn't recognized in the ConEMU terminal, check that WinPython is properly installed and the paths in env.bat are correct.
- If specific Python packages are missing, you may need to install them using pip within the ConEMU environment.
- If the working directory is not set correctly, ensure ConEMU.bat is placed in the desired directory before running.
- For permission issues, try running ConEMU.bat as an administrator.

# Conclusion

The ConEMU.bat provides a convenient way to launch a python environment ether for testing or as a custom environment to run python scripts. It avoids the need to worry about things like the python search path and may provide some security advantages. 

Tutorial Version: 0.1.0
Tutorial last updated 14-Aug-24
This tutorial is updated periodically. Check the GitHub repository for the most recent version.

# References

[1] WinPython. (n.d.). Home page. Retrieved August 14, 2024, from https://winpython.github.io/

## Additional references:
[2] Raybaut, P. (n.d.). Why WinPython? WinPython Documentation. https://winpython.github.io/#why-winpython
[3] Python Software Foundation. (n.d.). Python for Windows. https://www.python.org/downloads/windows/