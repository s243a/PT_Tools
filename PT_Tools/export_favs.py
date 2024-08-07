# This code assumes that you have the pywin32 module installed
# You can get it from https://pypi.org/project/pywin32/
# This code also assumes that you have the Internet Explorer bookmarks stored in the default location
# which is C:\\Users\\<username>\\Favorites
# This code will create a file called bookmarks.html in the same folder as this script
# You can then import this file to other browsers that support HTML format

import os
import datetime # for working with dates and times
#import win32com.client
# Create an Internet Explorer application object
#ie = win32com.client.Dispatch("InternetExplorer.Application")

global_defaults={
    'ADD_DATE':True,
    'ICON':True
}

# Get the current user name
username = os.environ.get("USERNAME")

# Get the path to the Favorites folder
favorites_path = os.path.join("C:\\Users", username, "Favorites")



out_dir = os.path.join("C:\\Users", username, "Documents")

# Loop through the Favorites folder and get the URLs and titles of the bookmarks

# #Create a HTML file to store the bookmarks
# Use generators instead. See: https://www.geeksforgeeks.org/generators-in-python/

import win32com.client
import win32api
import win32con
import win32ui
import tempfile
import win32gui
import base64
import win32com.client
from PIL import Image

 


#with tempfile.NamedTemporaryFile() as tmp:
def temp_bmp_icon_file(filename,*args):

    # Use the file name to get the icon location
    shell = win32com.client.Dispatch("WScript.Shell")
    icon = shell.GetIconLocation(filename)

    # Get the iconinfo tuple from the icon location
    iconinfo = win32gui.ExtractIconEx(icon[0], icon[1], 1, 1)

    # Create a bitmap object from the icon mask
    bmp = win32ui.CreateBitmapFromHandle(iconinfo[4])

    # Create a temporary file name for the bitmap file
    bmp_file, bmp_name = tempfile.mkstemp(suffix=".bmp")

    # Save the bitmap file
    bmp.SaveBitmapFile(win32ui.CreateDCFromHandle(0), bmp_name)

    # Close the icon handles
    win32gui.DestroyIcon(iconinfo[0])
    win32gui.DestroyIcon(iconinfo[4])
    return bmp_file

#Returns a temporary .ico file from a bmp file
def temp_ico_file(filename,**kw):
    #Get a temporary .bmp file repersenting the directory 
    #items icon. 
    with temp_bmp_icon_file(filename) as tmp_bmp:
        
        #filename = r'icon.bmp' # change this to your bitmap file path
        img = Image.open(tmp_bmp)
        tmp_ico=tempfile.TemporaryFile()    
        img.save(tmp_ico,'icon.ico') # change this to your desired ico file name
        return tmp_ico

def get_favicon_str(filename,**kw):
    with temp_bmp_icon_file(filename,**kw) as bmp_file:
        with temp_ico_file(bmp_file,**kw) as ico_file:
            return get_favicon_icon_str(filename)
def get_favicon_icon_str(filename):
    is_custom, icon_path = is_default_icon(co_file)
    if is_custom:
        with open(filename, 'rb') as f: # open the file in binary mode
            data = f.read() # read the file content as bytes
            b64 = base64.b64encode(data) # encode the bytes to base64
            b64 = b64.decode('ascii') # decode the base64 bytes to ascii string
            print(b64) # print the base64 string
            return filename

def get_dir_entry_args(dir_entry):   
  output={
    date:dir_entry.lstat().st_mtime
  }
  return output

def is_default_icon(filename):
    #filename = r'C:\Windows' # change this to your folder item path
    shell = win32com.client.Dispatch("WScript.Shell")
    icon = shell.GetIconLocation(filename) # returns a tuple of icon file and index
    if icon[0] == '': # if the icon file is empty, it means the folder item uses the default icon
        #return print('The icon for', filename, 'is the default icon.')
        return false, ""
    else: # otherwise, it means the folder item uses a custom icon
        #print('The icon for', filename, 'is a custom icon.')
        return true, icon

delta_indent="    "
indent=""
def increase_indent(indent):
    return indent+delta_indent


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                
out_path = os.path.join(out_dir, "bookmarks-" + now + ".html")

def read_hyperlink_fm_url_file(file_path):
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("URL="):
                url = line[4:].strip()
                return url

def write_bookmark_url(outfile,dir_entry,**kw):
    #with outfile as f:
        f=outfile
        indent=kw['indent']
        url=read_hyperlink_fm_url_file(dir_entry)
        f.write(indent+'<DT><A HREF=\"{}\"'.format(url))
        for atribute in ['ADD_DATE', 'LAST_MODIFIED', 'ICON']:
            if atribute in kw:
                f.write(' {}="{}"'.format(atribute,kw[atribute]))
        title=dir_entry.name[:-4]        
        f.write(">{}</A>\n".format(title))
    # Write the bookmarks as HTML links
def write_folder_heading(outfile,dir_entry,**kw):
        f=outfile
        indent=kw['indent']
        f.write(indent+'<DT><H3')
        for atribute in ['ADD_DATE', 'LAST_MODIFIED', 'PERSONAL_TOOLBAR_FOLDER', 'ICON']:
            if atribute in kw:
                f.write(' {}="{}"'.format(atribute,kw[atribute]))
        title=dir_entry.name #[:-4]
        f.write(">{}</H3>\n".format(title))            
def name_sort(value):
  # return the name attribute of the DirEntry object
  return value.name

def process_dir(dir_path_entry, **kw): #(path, file_action, dir_action, **kw):
    # Apply the given actions to files and directories in the given path
    result = []
    file_action=kw['file_action']
    dir_action=kw['dir_action']
    #Path = os.getcwd()

    # get an iterator of DirEntry objects
    entries = os.scandir(dir_path_entry)

    # sort the entries by their name
    sorted_entries = sorted(entries, key=name_sort)   
    for entry in sorted_entries: #os.scandir(dir_path_entry):
        if entry.is_file():
            # Apply the file_action to the entry
            result.append(file_action(entry, **kw))
        elif entry.is_dir():
            # Apply the dir_action to the entry
            result.append(dir_action(entry, **kw))
    return result 
def write_bookmark_folder(outfile,dir_entry,**kw):
    #with outfile as f:    
        f=outfile

        kw2=kw.copy()
        kw2['indent']=increase_indent(kw2['indent'])        
        write_folder_heading(f,dir_entry,**kw2)
        f.write(kw2['indent']+'<DL><p>\n')        

        kw3=kw2.copy()
        kw3['indent']=increase_indent(kw3['indent'])
        process_dir(dir_entry,**kw3)

        f.write(kw2['indent']+'</DL><p>\n')  
#def write_bookmarks(file_handle,in_file,out_file,**kw):
def write_header(f_hndl,**kw):
    #with f_hndl as f:
        f=f_hndl
        #if f.closed:
        #   f=open(f.path, "w")
        f.write("<!DOCTYPE NETSCAPE-Bookmark-file-1>\n")
        f.write("<META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=UTF-8\">\n")
        f.write("<TITLE>Bookmarks</TITLE>\n")
        f.write("<H1>Bookmarks</H1>\n")    
def write_bookmarks(**kw):

    if 'in_folder' not in kw:
        in_folder = os.path.join("C:\\Users", username, "Favorites")
    if 'out_folder' not in kw:
        if 'out_file' in kw:
            #TODO: use '.' as dir name if the following expression returns nothingor fails
            out_folder=os.dirname(kw['out_file'])
        else: 
            out_folder = os.path.join("C:\\Users", username, "Documents")
    if 'out_file' not in kw:
        now=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_file=now+".html"  
    if 'indent' not in kw:
        kw['indent']=''
    out_path=os.path.join(out_folder,out_file)
    f=open(out_path, "w+", encoding='utf-8')
    if 'file_action' not in kw:
        kw['file_action']=lambda dir_entry,**kw: write_bookmark_url(f,dir_entry,**kw)
    if 'dir_action' not in kw:
        kw['dir_action']=lambda dir_entry,**kw: write_bookmark_folder(f,dir_entry,**kw)
    write_header(f, **kw)
    process_dir(in_folder, **kw)
    f.write('</DL><p>\n')
    f.close()
#with open(file_path, "r") as f:   

write_bookmarks()

# Close the Internet Explorer application object
#ie.Quit()


# Print a message to indicate the completion of the task
print("Bookmarks exported to bookmarks.html")

