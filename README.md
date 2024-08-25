#PT_Tools

# Table of Contents
- [Introduction](#introduction)
- [Principal Tools](#principal-tools)
- [Helper Tools](#helper-tools)
- [In-Development Tools](#in-development-tools)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

#Intro

This project will contain a number of tools that work with [Pearltrees](https://www.pearltrees.com/), and some of these may be reusable on other social bookmarking sites. This project is being developed currently in windows but uses python which is cross platform. 

#Principal Tools

Currently, the principal scripts in this project are designed to scrape links from a webpage such as wikipedia, and then translate them into a format that can be imported into Pearltrees.com. As discussed in the tutorial:

[Tutorial_001_Importing Scraped Links %20Pearltrees%23.md](./Tutorials
/Tutorial_001_Importing%20Scraped%20Links%20(Pearltrees).md)

this is done using two python scripts:
1.  [pt_clip.py](./PT_Tools/pt_clip.py) creates an internet shortcut for each hyperlink and places it into a sub-directory of the Favorites folder, and
2. [export_favs.py](./PT_Tools/export_favs.py), which translates the Favorites folder into Netscape bookmark format so that it can be imported into Pearltrees. 

these tools are designed for simple html pages like Wikipedia and may not work on tools with a lot of dynamic content. Wikipedia is a useful site to scrape because the category system in Wikipedia can be used as a structure to organize links that one uploads into Pearltrees. 

#Helper Tools

## "Desktop_Cmd.bat"  and "ConEMU.bat"

There is also a helper tool that one can use to quickly set up their python scripting environment. As discussed in the tutorial:

[Tutorial_002 Setting Up a Python Environment with Bat Files.md](./Tutorials/Tutorial_002 Setting Up a Python Environment with Bat Files.md)


 It consists of two .bat files:
1. "[Desktop_Cmd.bat](./PT_Tools/Desktop_Cmd.bat)" - Sets the working directory to the location of this .bat file
2. "[ConEMU.bat](./WPy/ConEMU.bat)" - Set's up the rest of the Python environment (e.g. the location of the python interpret).

These scripts are developed to use WinPython but could be adapted to other versions of python. They aren't necessary. One could instead, just use environmental variables instead (which is what the python.org install does) or use another environment such as anaconda, that does this for you. 

However, if one takes one of these alternate approach then one must either make sure any scripts used are either in the python search path or python working directory.

## yt_LB_fix.py - Youtube linebreak Fix

Pearltrees allows one to make notes about things they found online. For videos, portions of a transcript may contain useful information that we need to remember at a later time. The script [yt_LB_fix.py](./PT_Tools/yt_LB_fix.py) edits the information on the clipboard to fix the line breaks in a copied portion of the YouTube transcript. Currently, it has a bug if the transcript is longer than an hour.

  

# In development tools

 ## phet.py (Pearltrees html export translater)

Pearltrees tells users that they own their own data but the export formats might be difficult for most users to work with. For instance, rather than an export of one's entire Pearltrees account, into NetScape Bookmark format, one might want to translate a given, so called Tree (for historical reasons, but is now more grid like), into a web page that contains only the links in that page and nothing deeper. 

Additionally, one might want do not only do this for a given so called Tree but also recurse into the sub-trees and create a separate web-page for each sub-tree. This project has a tool to do just that. The tool is called:

[phet.py](./PT_Tools/phet.py) -- Pearltrees HTML Export Translator

This tool is in early development. No warranties provided. Currently, there is not option to limit recursion depth, and all sub-trees are stored in an HTML file within the same folder. 

**Note:** I'm open to suggestions if anyone has a better name for this tool.

## Vid_TOC_scrap.py (Not yet on github)

This tools is not yet ready for github. The intent of this tool is to make video table of contents information easier to import into Pearltrees. 

# Requirements

[jaraco clipboard](https://pypi.org/project/jaraco.clipboard/)  - required for, pt_clip.py
[crate](https://pypi.org/project/crate/) - required for phet.py

# Usage

1. **pt_clip.py**: Run this script to scrape links from a web page and create internet shortcuts.
2. **export_favs.py**: Use this to convert the Favorites folder to Netscape bookmark format.
3. **yt_LB_fix.py**: Run this to fix line breaks in copied YouTube transcripts.

For detailed instructions, refer to our [tutorials](./Tutorials/).

# Contributing
This project is currently maintained by s243a ([Pearltrees](https://www.pearltrees.com/s243a), [GitHub](https://github.com/s243a)) with assistance from AI tools like Claude.AI and Copilot. We welcome contributions from the community, including bug reports, feature requests, and code contributions. If you'd like to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

# License

To be determined. We are open to suggestions for an appropriate open-source license. If you have experience with open-source licensing and would like to recommend a license that aligns with the goals of this project, please open an issue to discuss.