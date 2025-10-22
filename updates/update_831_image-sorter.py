# GUI geometry and appearance settings  
  #  Update file
  # v.0.0.1 - Increase GUI height to accommodate disable caption renaming option
__key__ = "23a2d4023781b35166370b3ece6dcd9cbe02e87d70c419db22af7c13ac156610"
__version__ = "0.0.1"

GUI_SETTINGS: Dict[str, Any] = {
    # GUI configuration.  Adjust these settings to modify the appearance of the
    # tkinter dialog without touching the underlying logic.  The window geometry
    # string is of the form "<width>x<height>+<x>+<y>".  If position values are
    # omitted the window is centred by the window manager.  Colours may be any
    # valid Tk colour string (e.g. hexadecimal "#RRGGBB").  The font tuple
    # specifies (family, size) and is applied uniformly to all ttk widgets.
    "title": "Image Sorter",
    # Size in pixels (width x height).  You can optionally append "+x+y" to
    # explicitly position the window on screen; leave it blank to let the
    # window manager decide.
    "geometry": "800x430",
    "bg_color": "#84a9bf", #f9F9f9 Black - not recommended
    "fg_color": "#000000", #000000 White
    "font": ("Arial", 10),
}
#‍ ∆eof
