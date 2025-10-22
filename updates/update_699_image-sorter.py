# Arg_Defintions - Defines GUI / TUI / Argument behaviours using ArgSpec class  
  #  Update file
  # v.0.0.1 - Added disable renaming option
__key__ = "3fb38e7d31658e4932c074f75c79a2784c04820830b5b68c426ca15d8618603d"
__version__ = "0.0.1"

# Argument definitions.  To add, remove or modify an argument, edit this.
# ArgSpec is used to drive the CLI, GUI and TUI prompts.
ARG_DEFINITIONS: List[ArgSpec] = [
    ArgSpec(
        name="source",
        cli_flags=None,
        type=Path,
        default=_default_source,
        help="Directory of images (default: current working directory)",
        gui=True,
        gui_label="Source directory",
        gui_widget="dir",
        interactive_prompt="Source directory",
        nargs="?",
    ),
    ArgSpec(
        name="dest",
        cli_flags=["--dest"],
        type=Path,
        default=_default_dest,
        help=(
            "Destination root (default: <cwd>/images; falls back to <cwd>/Processed_Images if equal to source)"
        ),
        gui=True,
        gui_label="Destination directory",
        gui_widget="dir",
        interactive_prompt="Destination directory (leave blank for default)",
    ),
    ArgSpec(
        name="log",
        cli_flags=["--log"],
        type=bool,
        default=False,
        help="Enable logging of output paths",
        gui=True,
        gui_label="Enable logging",
        gui_widget="checkbox",
        interactive_prompt="Enable logging? (y/N)",
        action="store_true",
    ),
    ArgSpec(
        name="clip_batch",
        cli_flags=["--clip-batch"],
        type=int,
        default=32,
        help="Batch size for CLIP image encoding",
        gui=True,
        gui_label="Clip batch size",
        gui_widget="entry",
        interactive_prompt="Clip batch size",
    ),
    ArgSpec(
        name="dry_run",
        cli_flags=["--dry-run"],
        type=bool,
        default=False,
        help="Do not move files (dry run)",
        gui=True,
        gui_label="Dry run (no moves)",
        gui_widget="checkbox",
        interactive_prompt="Dry run (no moves)? (y/N)",
        action="store_true",
    ),
    ArgSpec(
        name="save_map",
        cli_flags=["--save-map"],
        type=Path,
        default=_default_save_map,
        help="Optional path to write Destinations JSON",
        gui=True,
        gui_label="Save map to",
        gui_widget="savefile",
        interactive_prompt="Save map to file (leave blank to skip)",
    ),
    ArgSpec(
        name="no_captioning",
        cli_flags=["--no-captioning"],
        type=bool,
        default=False,
        help="Do not rename files",
        gui=True,
        gui_label="No Captioning (considerably faster, slightly less accurate categorization)",
        gui_widget="checkbox",
        interactive_prompt="considerably faster, slightly less accurate categorization)? (y/N)",
        action="store_true",
    ),
    ArgSpec(
        name="disable_renaming",
        cli_flags=["--disable-renaming"],
        type=bool,
        default=False,
        help="Do not rename files",
        gui=True,
        gui_label="Disable caption renaming",
        gui_widget="checkbox",
        interactive_prompt="Disable caption renaming? (y/N)",
        action="store_true",
    ),
    ArgSpec(
        name="monitor",
        cli_flags=["--monitor"],
        type=bool,
        default=True,
        help="Monitor directory for new files",
        gui=True,
        gui_label="Monitor directory for new files",
        gui_widget="checkbox",
        interactive_prompt="Monitor directory for new files? (y/N)",
        action="store_true",
    ),
    ArgSpec(
        name="AutoUpdate",
        cli_flags=["--AutoUpdate"],
        type=bool,
        default=False,
        help="Get the latest Updates as they are released",
        gui=True,
        gui_label="Get the latest Updates as they are released",
        gui_widget="checkbox",
        interactive_prompt="Get the latest Updates as they are released? (y/N)",
        action="store_true",
    ),

    # Select the underlying CLIP model used for zero‑shot classification.  A
    # predefined list of models is presented in the GUI via a dropdown, but
    # users may also type in other model names.  The default matches the
    # original behaviour (ViT‑B/32).
    ArgSpec(
        name="clip_model",
        cli_flags=["--clip-model"],
        type=str,
        default="ViT-B/32",
        help="Name of the CLIP model used for zero-shot classification (e.g. ViT-B/32)",
        gui=True,
        gui_label="Clip model",
        gui_widget="dropdown",
        interactive_prompt="Clip model (choose from known models or enter manually)",
        choices=DEFAULT_CLIP_MODELS,
    ),
]
#‍ ∆eof
