# This code assumes that you have the pywin32 module installed
# You can get it from https://pypi.org/project/pywin32/
# This code also assumes that you have the Internet Explorer bookmarks stored in the default location
# which is C:\\Users\\<username>\\Favorites
# This code will create a file called bookmarks.html in the same folder as this script
# You can then import this file to other browsers that support HTML format

import os
import datetime # for working with dates and times
import warnings
import re
#import win32com.client
# Create an Internet Explorer application object
#ie = win32com.client.Dispatch("InternetExplorer.Application")

global_defaults={
    'ADD_DATE':True,
    'ICON':True
}
# TODO: In the future, move this to a utility module with proper deprecation handling
def deprecated(func):
    """
    Decorator to mark functions as deprecated.
    It will result in a warning being emitted when the function is used.
    """
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is deprecated and will be removed in a future version. "
            f"Use read_url_file_data instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


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
    #is_custom, icon_path = is_default_icon(co_file)
    is_custom, icon_path = is_default_icon(filename)
    if is_custom and icon_path:
        try:
            with open(icon_path, 'rb') as f:  # open the icon file in binary mode
                data = f.read()  # read the file content as bytes
                b64 = base64.b64encode(data)  # encode the bytes to base64
                b64 = b64.decode('ascii')  # decode the base64 bytes to ascii string
                return f"data:image/x-icon;base64,{b64}"  # return as data URL
        except Exception as e:
            print(f"Error reading icon file {icon_path}: {e}")
    return None  # return None if not a custom icon or on error

def get_dir_entry_args(dir_entry):   
  output={
    'date':dir_entry.lstat().st_mtime
  }
  return output

def is_default_icon(filename):
    #filename = r'C:\Windows' # change this to your folder item path
    shell = win32com.client.Dispatch("WScript.Shell")
    icon = shell.GetIconLocation(filename) # returns a tuple of icon file and index
    if icon[0] == '': # if the icon file is empty, it means the folder item uses the default icon
        #return print('The icon for', filename, 'is the default icon.')
        return False, ""
    else: # otherwise, it means the folder item uses a custom icon
        #print('The icon for', filename, 'is a custom icon.')
        return True, icon

delta_indent="    "
indent=""
def increase_indent(indent):
    return indent+delta_indent


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")                
out_path = os.path.join(out_dir, "bookmarks-" + now + ".html")

@deprecated
def read_hyperlink_fm_url_file(file_path):
    #with open(file_path, "r") as f:
    #    for line in f:
    #        if line.startswith("URL="):
    #            url = line[4:].strip()
    #            return url
    try:
        # Use latin-1 encoding which can handle any byte value
        with open(file_path, "r", encoding='latin-1') as f:
            for line in f:
                if line.startswith("URL="):
                    url = line[4:].strip()
                    return url
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return None
def decode_url_encoded_title(title):
    """
    Decode URL-encoded characters in a title based on the specific encoding used.
    
    Args:
        title: The encoded title string (filename)
    
    Returns:
        str: The decoded title suitable for Netscape bookmark format
    """
    # Remove the .url extension if present
    if title.lower().endswith('.url'):
        title = title[:-4]
    
    # Reverse the specific replacements that were made during encoding
    decoded = (title.replace("%3A", ":")
                   .replace("%2F", "/")
                   .replace("_star_", "*")  # Support current custom encoding
                   .replace("%2A", "*")     # Support standard URL encoding
                   .replace("%22", '"')
                   .replace("%3F", "?"))
        
    return decoded
def read_url_file_data(file_path):
    """
    Read a .url file and extract both the URL and title if available.
    
    Args:
        file_path: Path to the .url file
        
    Returns:
        tuple: (url, title) where title may be None if not found
    """
    url = None
    title = None
    
    try:
        with open(file_path, "r", encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line.startswith("URL="):
                    url = line[4:]
                # Check for various possible title fields
                elif line.startswith("TITLE=") or line.startswith("Title="):
                    title = line.split('=', 1)[1]
                elif line.startswith("BASEURL="):  # Some .url files use this for title
                    if not title:  # Only use as fallback
                        title = line.split('=', 1)[1]
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return url, title

def escape_html_content(text):
    """
    Escape special characters for HTML content while preserving existing entities.
    
    This function uses regular expressions to only escape ampersands that aren't
    already part of valid HTML entities, and also escapes < and > characters.
    
    Valid HTML entity patterns that will be preserved:
    1. Named entities: &name; (e.g., &amp;, &lt;) - limited to 1-8 alpha chars
    2. Decimal entities: &#number; (e.g., &#38;) - limited to 1-7 digits
    3. Hex entities: &#xhex; (e.g., &#x26;) - limited to 1-6 hex digits
    
    Args:
        text: Text to escape
        
    Returns:
        str: HTML-escaped text with existing entities preserved
    """
    # Replace ampersands that aren't part of valid HTML entities
    text = re.sub(r'&(?!(#[0-9]{1,7};|#x[0-9a-fA-F]{1,6};|[a-zA-Z]{1,8};))', '&amp;', text)
    
    # Escape < and > normally
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    
    return text

def write_bookmark_url(outfile,dir_entry,**kw):

    f=outfile
    indent=kw['indent']

    # Check if it's a .url file
    if not dir_entry.name.lower().endswith('.url'):
        return None  # Skip non-URL files
    
    # Use dir_entry.path instead of dir_entry directly
    #url=read_hyperlink_fm_url_file(dir_entry)
    #url = read_hyperlink_fm_url_file(dir_entry.path)
    url, file_title = read_url_file_data(dir_entry.path)

    # Check if we got a URL
    if url is None:
        print(f"Warning: Could not read URL from {dir_entry.path}")
        return None

    # Determine the title using the following order:
    # 1. Title from the .url file if available
    # 2. Decoded filename if no title in file
    if file_title:
        title = file_title  # Use title from file
    else:
        # Get the title from filename and decode it
        raw_title = dir_entry.name[:-4]
        title = decode_url_encoded_title(raw_title)

    # HTML escape the title (regardless of its source)
    title = escape_html_content(title)
    
    # Safely escape the URL for HTML attributes (quotes matter here)
    safe_url = url.replace('"', "&quot;")


    f.write(indent+f'<DT><A HREF="{safe_url}"')  # Using the safe_url
    for atribute in ['ADD_DATE', 'LAST_MODIFIED', 'ICON']:
        if atribute in kw:
            f.write(' {}="{}"'.format(atribute,kw[atribute]))
  
    f.write(">{}</A>\n".format(title))
    return True

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

