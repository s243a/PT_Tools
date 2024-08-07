#PT_Tools
This project will contain a number of tools that work with Pearltrees, and some of these may be reusable on other social bookmarking sites. Currently these tools only work on Windows. 

The principle two intial scripts are:
pt_clip.py
export_favs.py

One can use this to for example scrape the url on a Wikipedia page, but it will also work on many other sites. I mention Wikipedia specifically because one might want to use Wikipedia categories for the organization of their bookmarks on Pearltrees.

First highlight the urls you want on the web page and then copy them to the clipboard. 

Then run the command,
`python pt_clip.py

This will read the clipboard and save each url in a bookmark file at:

"C:\Users\johnc\Favorites"

Modify the code to the location of you Favorites folder:
https://github.com/s243a/PT_Tools/blob/d984cb9047276a389a1d2d13953bb05c543be1c9/PT_Tools/pt_clip.py#L35C16-L35C41

After pt_clip.py has copied the bookmarks to a folder within your favourites folder you then export your favourites folder to netscape bookmark format. This is done with the command:

`python export_favs.py

This command will export the netscape bookmark file to:

"C:\Users\username\Documents"

where username is the actual user name of the user running windows. You'll have to navigate to this folder because it likely won't be your normal "Documents" folder.

Now that you generated a Netwscape bookmark file at:

"C:\Users\username\Documents"

you can import this file into pearltrees using the import bookmarks feature. When doing this I select, "Import chrome bookmarks".  

I've also included a command called, "Deskto_Cmd.bat", this will create a custom terminal, that sets python to, "Win Python 64". This is optional, Other alternatives:

- set your can use environmental variables to set the python command to your Favorites python implementation.
- use some kind of environment containing the python command like Anaconda. 