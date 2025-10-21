# -*- encoding: utf-8 -*-
from __future__ import annotations
import warnings, logging as log
import datetime as dt
from datetime import datetime
import os
import re
import sys
import shutil
import unicodedata
import importlib.util
import inspect
import time
import argparse
import json
import textwrap
import linecache
import hashlib
import requests
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Sequence
import torch
import clip
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from dataclasses import dataclass, field

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Enable symlink warning suppression
from transformers import BlipProcessor, BlipForConditionalGeneration, logging as hf_logging

# Create the config folder path inside that home directory
# These constants shared between multiple functions.
if getattr(sys, 'frozen', False):  # Running as a PyInstaller EXE
    script_home = Path(sys.executable).parent
else:
    script_home = Path(__file__).resolve().parent

config_dir = script_home / "config"
updates_dir = script_home / "updates"
config_dir.mkdir(exist_ok=True)

# Point to the prefs file inside config
prefs_file = config_dir / "image-sorter.prefs.json"

DELTA_TOKEN = "∆eof"
pair_spaced_comment = "  #  Update file"
ZWJ = "\u200d"

def ensure_first_line_trailing_spaces(lines):
    """Ensure first line ends with exactly two trailing spaces (append if needed)."""
    if not lines:
        return ["  "]
    first = lines[0]
    # remove existing trailing spaces, then add exactly two
    stripped = first.rstrip("\n").rstrip("\r")
    if stripped.endswith("  "):
        return lines
    # preserve any newline character from original first line if present
    nl = ""
    if first.endswith("\r\n"):
        nl = "\r\n"
    elif first.endswith("\n"):
        nl = "\n"
    lines[0] = stripped + "  " + nl
    return lines

def has_pair_spaced_comment(lines):
    return any(re.match(r"^ {2}# {2}", ln) for ln in lines)

def has_delta_token(lines):
    return any("#" in ln and DELTA_TOKEN in ln for ln in lines)

def insert_pair_spaced_comment_after_first(lines):
    # insert the exact 4-space comment after the first line
    if not lines:
        lines.append(pair_spaced_comment + "\n")
    else:
        # place it after first non-empty line to avoid being inside shebang sometimes
        insert_at = 1
        lines.insert(insert_at, pair_spaced_comment + "\n")
    return lines

def append_delta_comment(lines):
    # append a comment with the delta token at end (on its own line)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1] + "\n"
    lines.append(f"#{ZWJ} {DELTA_TOKEN}\n")
    return lines

def process_src(raw_src: str) -> str:
    # normalize newlines to \n for easier handling, but preserve later when writing
    raw = raw_src.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.splitlines(True)  # keepends=True so we can preserve EOLs

    # 1) ensure first line ends with exactly two spaces
    lines = ensure_first_line_trailing_spaces(lines)

    # 2) ensure a 4-space-leading comment exists
    if not has_pair_spaced_comment(lines):
        lines = insert_pair_spaced_comment_after_first(lines)

    # 3) ensure delta token exists in a comment somewhere
    if not has_delta_token(lines):
        # try to add it as an inline comment to the last non-empty line if that is safe,
        # otherwise append a new comment line
        # We'll append a new comment line to avoid modifying code lines.
        lines = append_delta_comment(lines)

    # return combined text (use \n endings)
    return "".join(lines)

class Updatable:
    def __init__(self, *, code: Optional[str] = None,
                 func: Optional[Callable[[Dict[str, Any]], Any]] = None, **assignments):

        f = inspect.currentframe().f_back
        info = inspect.getframeinfo(f)
        filename = info.filename
        line = info.lineno
        prev_line = linecache.getline(filename, line - 1).rstrip()

        stem = os.path.splitext(os.path.basename(
            sys.executable if getattr(sys, 'frozen', False) else info.filename
        ))[0]

        # debug flags
        dbg = f"{stem}_debug"
        dbg_verbose = f"{stem}_debugVerbose"
        debug = dbg in os.environ or dbg in globals()
        debug_verbose = dbg_verbose in os.environ or dbg_verbose in globals()

        # show or capture debug info
        if debug == "segmentID":
            print(f"Updatable → {{\n  {prev_line}\n  {stem}:{line}\n}}", end="")

        # propagate any passed-in globals
        for k, v in assignments.items():
            globals()[k] = v

        # patch file path
        path = os.path.join("updates", f"update_{line}_{stem}.py")
        if os.path.exists(path):
            if code: 
                key = code.encode("utf-8")
                hash_digest = hashlib.sha256(key).hexdigest()
            with open(path, encoding="utf-8") as f:
                src = f.read()
                if hash_digest not in src:
                    print(f"{hash_digest}")
                    print(f"Update file: {path}\n does not contain a valid key. Reverting to default.")
                    try:
                        exec(code, globals())
                        return
                    except Exception as e:
                        print(f"Error: {e} in default code on line {line}")
                        raise SystemExit(0)
            if self._ok(src):
                try:
                    exec(src, globals())
                    if debug:
                        print(f"Applied update file: {path}\n")
                except Exception as e:
                    print(f"Error in update file: {path}\n {e}\n")
            elif debug:
                print(f"Skipped invalid patch {path}")
            return

        if code:
            # always summarize number of lines
            if debug == "segment":
                print(" - segment contains", int(len(code.splitlines()) - 1), "lines\n")

            # if verbose debugging is enabled, export code instead of printing
            if debug_verbose:
                try:
                    export_dir = script_home / "exports"
                    export_dir.mkdir(exist_ok=True)
                    export_path = export_dir / f"update_{line}_{stem}.py"

                    # --- Compute hash of the original code ---
                    key = code.encode("utf-8")
                    hash_digest = hashlib.sha256(key).hexdigest()
                    key_info = f'__key__ = "{hash_digest}"\n'
                    ver_info = '__version__ = "0.0.0"\n'

                    # --- Insert hash line + ver between prev_line and code ---
                    export = prev_line + "\n" + key_info + ver_info + code

                    # --- Process and write to disk ---
                    processed_code = process_src(export)
                    with export_path.open("w", encoding="utf-8", newline="\n") as f_out:
                        f_out.write(processed_code)

                except Exception as e:
                    print(f"[Updatable] Failed to export debug segment: {e}")

            else:
                # normal debug just prints code
                if debug == "code":
                    print(f"{code}\n")

            # execute the given code segment
            try:
                exec(textwrap.dedent(code).lstrip("\n"), globals())
            except Exception as e:
                print(f"{e} line: {line}")

        if func:
            func(globals())

    def _ok(self, s: str) -> bool:
        """4 stealthy syntax fingerprints including a zero-width joiner"""
        L = s.splitlines()
        return (
            L
            and L[0].endswith("  ")                        # ① two trailing spaces on first line
            and any(re.match(r"^ {2}# {2}", x) for x in L) # ② comment indented 2 spaces and comment initiator suffixed by 2 spaces
            and re.search(r"#.*∆eof", s)                   # ③ Unicode delta token
            and "\u200d" in s                              # ④ hidden zero-width joiner
        )

# ensure network connectivity before script run.
Updatable(code=r"""
import socket
def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

no_internet = True

while no_internet:
    if not check_internet():
        input("Warning - Internet connection required. Connect to the internet to continue.")
    else:
        no_internet = False
        continue
""")

# _parse_version_key - helper for Github self Updater
Updatable(code=r"""
def _parse_version_key(value: str) -> Tuple[str, str]:
    # Extracts the version and key from a manifest value.

    # The manifest entries are strings like "ver:1.2.3 key:abc".  This
    # helper splits the string on whitespace and returns a tuple of
    # (version, key).  If either element is missing an empty string is
    # returned for that slot.

    # Args:
    #    value: The raw value from the manifest.

    # Returns:
    #    A tuple (version, key) where each element may be an empty string.
    
    version = ""
    key = ""
    if isinstance(value, str):
        for part in value.split():
            if part.startswith("ver:"):
                version = part.split(":", 1)[1]
            elif part.startswith("key:"):
                key = part.split(":", 1)[1]
    return version, key
""")

# _is_version_newer - helper for Github self Updater
Updatable(code=r"""
def _is_version_newer(remote: str, local: str) -> bool:
    if not local:
        return True
    try:
        remote_parts = [int(p) for p in remote.split(".")]
        local_parts = [int(p) for p in local.split(".")]
    except (ValueError, AttributeError):
        # Non‑numeric or malformed version strings cause a refresh
        return True
    # pad shorter list with zeros
    length = max(len(remote_parts), len(local_parts))
    remote_parts += [0] * (length - len(remote_parts))
    local_parts += [0] * (length - len(local_parts))
    return remote_parts > local_parts
""")

# version ID - github self update helper
Updatable(code=r"""
def _extract_version_from_file(content: str) -> str:
    match = re.search(r'__version__\s*=\s*["\']([\d.]+)["\']', content)
    return match.group(1) if match else "0.0.0"
""")

# github self updater. Opt-In.
Updatable(code=r"""
def fetch_all_updates_from_github(
    repo_owner: str,
    repo_name: str,
    branch: str = "main",
    subdir: str = "updates",
    config_dir: str = "./config",     # provide your config_dir path
    updates_dir: str = "./updates",   # provide your updates_dir path
):
    
    # Fetch and cache self‑update scripts from GitHub.

    # The updater reads a version manifest (``update_ver.json``) from the
    # specified repository and determines which update files to download based on
    # version information stored in a local ``updates.json``.  Only files with
    # newer versions or missing entries are retrieved, and the check runs at
    # most once per day.

    # Args:
    #    repo_owner: GitHub username or organisation owning the repo.
    #    repo_name: Name of the GitHub repository.
    #    branch: Branch from which to fetch updates (default "main").
    #    subdir: Subdirectory within the repository containing updates and the
    #        ``update_ver.json`` manifest (default "updates").
    #    config_dir: Local directory to store ``updates.json`` and
    #        ``update_check.txt`` (default ``./config``).
    #    updates_dir: Local directory in which to write downloaded update
    #        scripts (default ``./updates``).

    # Returns:
    #    None
 
    # Ensure directories exist
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(updates_dir, exist_ok=True)

    # Compute paths for the timestamp and version metadata
    last_check_file = os.path.join(config_dir, "update_check.txt")
    updates_json = os.path.join(config_dir, "updates.json")
    prefs_file = Path(config_dir) / "image-sorter.prefs.json"

    saved_prefs: Dict[str, Any] = {}
    if prefs_file.is_file():
        try:
            with open(prefs_file, "r", encoding="utf-8") as pf:
                saved_prefs = json.load(pf) or {}
        except Exception:
            saved_prefs = {}

    Auto_Update = bool(saved_prefs.get("AutoUpdate", False))

    if not Auto_Update:
       return
    # Load existing version data (filename -> manifest string)
    if os.path.exists(updates_json):
        try:
            with open(updates_json, "r", encoding="utf-8") as f:
                local_versions: Dict[str, str] = json.load(f)
        except (json.JSONDecodeError, IOError):
            local_versions = {}
    else:
        local_versions = {}

    # Skip if already checked today
    if os.path.exists(last_check_file):
        try:
            with open(last_check_file, "r", encoding="utf-8") as f:
                last_check_str = f.read().strip()
            last_check_date = dt.datetime.strptime(last_check_str, "%Y-%m-%d").date()
            if last_check_date == dt.date.today():
                return
        except (ValueError, IOError):
            # ignore bad date and proceed
            pass

    # Build remote URLs for the manifest and raw files
    raw_base = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{subdir}"
    manifest_url = f"{raw_base}/update_ver.json"

    fetched_any = False
    try:
        # Download and parse the remote version manifest
        resp = requests.get(manifest_url, timeout=10)
        resp.raise_for_status()
        manifest = resp.json()  # Expecting a dict of {filename: "ver:X.Y.Z key:..."}

        # Iterate through each declared update file
        for filename, remote_meta in manifest.items():
            # Only consider Python update scripts
            if not isinstance(filename, str) or not filename.endswith(".py"):
                continue

            remote_version, remote_key = _parse_version_key(remote_meta)
            local_meta = local_versions.get(filename, "")
            # Extract local version (if present) from the stored meta value
            local_version = ""
            if isinstance(local_meta, str):
                local_version, _ = _parse_version_key(local_meta)

            # Determine whether to update
            if filename not in local_versions or _is_version_newer(remote_version, local_version):
                # Download the actual update file
                file_url = f"{raw_base}/{filename}"
                try:
                    content_resp = requests.get(file_url, timeout=10)
                    content_resp.raise_for_status()
                    content = content_resp.text
                    remote_version = _extract_version_from_file(content)
                except requests.RequestException:
                    # Skip this file if it cannot be downloaded
                    print(f"[Update Check] Failed to download {filename}")
                    continue

                local_path = os.path.join(updates_dir, filename)
                with open(local_path, "w", encoding="utf-8") as out:
                    out.write(content)

                # Update the metadata for this file
                local_versions[filename] = remote_meta
                print(f"[Fetched] {filename} (ver {remote_version})")
                fetched_any = True

        # Persist updated metadata if any files were fetched
        if fetched_any:
            with open(updates_json, "w", encoding="utf-8") as f:
                json.dump(local_versions, f, indent=2)

        # Record the timestamp of this check
        with open(last_check_file, "w", encoding="utf-8") as f:
            f.write(dt.date.today().isoformat())

    except requests.RequestException as e:
        print(f"[Startup Update Check] Failed to fetch update manifest: {e}")
""")

# Update source target
Updatable(code=r"""
fetch_all_updates_from_github(
    repo_owner="T3RRYT3RR0R",
    repo_name="Image-Sorter",
    branch="main",
    subdir="updates",
    config_dir=config_dir,
    updates_dir=updates_dir,
)
""")

# Suppress warnings from transformers and other libraries
Updatable(code="""
log.captureWarnings(True)
""")

# ModelLoadError class + ModelLoadAbort Alias
Updatable(code=r"""
# =============================================================================
# Custom Exception used to indicate repeated failures when downloading a model.
# When a CLIP model is being loaded from the Hugging Face hub it may need to
# download large weights from S3.  Occasionally the underlying S3 client will
# emit an error like ``Fatal Client Error: s3::get_range api call failed: Request
# failed after 5 retries`` multiple times before eventually succeeding.  To
# prevent the application from hanging indefinitely, we watch for this message
# during model loading and abort after it has been seen more than a handful of
# times.  ``ModelLoadError`` is raised in such a case so that the caller can
# reset the GUI and prompt the user to select a different model.
class ModelLoadError(Exception):
    # Raised when repeated S3 retrieval errors occur while loading a model.
    # This Exception is used both for aborting model loads in interactive
    # mode (where the GUI is reset to allow the user to pick a different
    # model) and in CLI mode (where the program exits with a non‑zero
    # status).  Historically the code raised ``ModelLoadError``, but
    # separate handling for aborts was needed to avoid raising from
    # inside logging handlers.  To keep the public API stable, this
    # single Exception class is now reused for both cases.
    pass

# Alias ``ModelLoadAbort`` to ``ModelLoadError`` for backward compatibility.
ModelLoadAbort = ModelLoadError
""")

# S3 error tracker
Updatable(code=r"""
# --- Count specific S3 retry errors without raising from inside logging ---
class _S3RetryCounter(log.Filter):
    # Counts lines like:
    # 'Fatal Client Error: s3::get_range api call failed: Request failed after N retries'
    # Never raises Exceptions. The decision to abort is made by the caller after the guarded operation.

    def __init__(self, threshold: int = 3) -> None:
        super().__init__()
        # Precompile a pattern that matches the s3 get_range retry errors irrespective of prefixes or timestamps.
        self.pat = re.compile(r"Fatal Client Error:\s*s3::get_range.*after\s+\d+\s+retries", re.IGNORECASE)
        self.threshold = threshold
        self.count = 0

    def filter(self, record: log.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if msg and self.pat.search(msg):
            self.count += 1
        # Always allow the record to be logged
        return True
""")

# S3 error context handler
Updatable(code=r"""
# --- Context manager to install a temporary logging handler that counts S3 retry errors ---
class s3_error_watch:
    # Context manager that installs a temporary handler+filter counting S3 get_range failures.
    # It never raises from inside logging. The caller checks the returned counter after the
    # guarded operation to decide whether to abort.
    def __init__(self, logger: Optional[log.Logger] = None, level: int = log.ERROR, threshold: int = 3):
        self.logger = logger or log.getLogger()
        self.level = level
        self.filter = _S3RetryCounter(threshold=threshold)
        self.handler = log.StreamHandler(stream=sys.stderr)
        self.handler.setLevel(level)
        self.handler.addFilter(self.filter)
        self._installed = False
        self._old_level = None

    def __enter__(self) -> _S3RetryCounter:
        # Ensure the logger is configured to propagate messages at least at the desired level
        if self.logger.level > self.level or self.logger.level == 0:
            # Save previous level and set a lower one temporarily
            self._old_level = self.logger.level
            self.logger.setLevel(min(self.logger.level or self.level, self.level))
        self.logger.addHandler(self.handler)
        self._installed = True
        return self.filter

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._installed:
            try:
                self.logger.removeHandler(self.handler)
            except Exception:
                pass
        if self._old_level is not None:
            try:
                self.logger.setLevel(self._old_level)
            except Exception:
                pass
        # Do not suppress Exceptions
        return False
""")

# Clip base model selectable options
Updatable(code=r"""
    # List of known CLIP model names for zero‑shot classification.  These
    # correspond to available model identifiers in the OpenAI/CLIP library.  The
    # GUI will present these as dropdown choices, but users may still type in
    # alternative model names if desired.
    DEFAULT_CLIP_MODELS: Sequence[str] = (
        # OpenAI CLIP models (available via the `clip` library)
        "ViT-B/32", # preferred model for performance + accuracy
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        # Non-standard Hugging Face zero‑shot classifiers.  These entries can be selected
        # directly from the dropdown; they are loaded via the Transformers API.
        # Users may also enter a custom model identifier if they wish to use a different
        # model.
        # When used within .venv models are installed under lib\site-packages
        #  - Note - not all models are guaranteed to work correctly.
        #           some models may require additional modules to be downloaded
        # Use the full identifier to ensure the correct model
        # checkpoint is downloaded.  Additional Hugging Face
        # model identifiers may be added here.
        "facebook/metaclip-2-worldwide-huge-quickgelu",
        "facebook/metaclip-h14-fullcc2.5b",
        # ZSC Models tested and found to be unfit for General-use application:
        # "google/siglip-base-patch16-512",
    )
""")

# detect if CLIP model architecture is HF
Updatable(code=r"""
# Helper to determine if a loaded Hugging Face model corresponds to a standard CLIP architecture.
# Standard CLIP models (e.g. ``CLIPModel``) support automatic multi‑device and device map
# handling out of the box.  Other models (e.g. MetaCLIP variants) may not accept device
# dictionaries and therefore must be placed entirely on a single device for inference.
def _is_standard_clip_model(model: Any) -> bool:
    # Return True if the given Transformers model is a standard CLIP architecture.

    # The check is based on the model configuration's ``architectures`` field.  If any of
    # the declared architectures contain ``CLIPModel`` then the model is treated as a
    # standard CLIP.  On any errors or when the field is absent, the function returns
    # False, indicating the model should be handled as non‑standard.

    # Parameters
    # ----------
    # model: Any
    #    The Hugging Face model instance to inspect.

    # Returns
    # -------
    # bool
    #    True if the model appears to be a standard CLIP model, otherwise False.
 
    try:
        cfg = getattr(model, "config", None)
        archs = getattr(cfg, "architectures", None)
        if archs:
            for arch in archs:
                # check for exact match or substring match to CLIPModel
                if isinstance(arch, str) and "CLIPModel" in arch:
                    return True
    except Exception:
        pass
    return False
""")

# GUI Config via Argspec Class
Updatable(code=r"""
# -----------------------------------------------------------------------------
# Argument specification and GUI configuration
#
# To make the CLI, tkinter GUI and text‑based (TUI) prompts easy to extend and
# maintain, we define a single data structure describing all supported
# arguments.  Each entry contains enough information to build the
# `argparse.ArgumentParser`, generate the tkinter widgets and drive the
# interactive prompts.  If you wish to add, modify or remove an argument,
# simply update the `ARG_DEFINITIONS` list rather than editing multiple code
# locations.  The `GUI_SETTINGS` dictionary controls the appearance of the
# tkinter window (size, position, colours and font).
@dataclass
class ArgSpec:
    # Specification for a single command‑line/GUI argument.

    # Attributes
    # ----------
    # name: str
    #    The attribute name used on the argparse.Namespace (e.g. "source").
    # cli_flags: Optional[Sequence[str]]
    #    A sequence of command‑line flags.  For positional arguments specify
    #    `None` and set `nargs` accordingly.
    # type: Any
    #    The Python type to cast the argument to (e.g. ``Path``, ``int``, ``bool``).
    #    For boolean flags this should generally be ``bool``.
    # default: Any
    #    The default value used for both CLI and GUI.  If this is a callable it
    #    will be invoked (without arguments) to compute the default at runtime.
    # help: str
    #    A help string used for CLI help output.
    # gui: bool
    #    Whether to include this argument in the tkinter GUI / TUI prompts.
    # gui_label: str
    #    A human‑readable label shown next to the GUI widget.
    # gui_widget: str
    #    The type of widget to use in the GUI; one of ``'entry'`` (free‑text
    #    input), ``'checkbox'`` (boolean), ``'dir'`` (directory picker), or
    #    ``'savefile'`` (file picker for save paths).
    # interactive_prompt: str
    #    The prompt used when falling back to a text‑based interactive mode.
    # nargs: Optional[str]
    #    Passed through to ``argparse.add_argument``.  Only used for positional
    #    arguments (e.g. ``nargs='?'`` allows omitting the argument entirely).
    # action: Optional[str]
    #    Passed through to ``argparse.add_argument``.  Generally used for
    #    boolean flags such as ``'store_true'``.

    name: str
    cli_flags: Optional[Sequence[str]]
    type: Any
    default: Any
    help: str
    gui: bool = True
    gui_label: str = ""
    gui_widget: str = "entry"
    interactive_prompt: str = ""
    nargs: Optional[str] = None
    action: Optional[str] = None
    # For widgets that present a fixed list of choices (e.g. dropdowns),
    # this optional sequence enumerates the permitted values.  The GUI
    # will present these as selectable options, but the user may still
    # enter arbitrary text unless the widget state is set to readonly.
    choices: Optional[Sequence[str]] = None

def _default_source() -> Optional[Path]:
    # Default for the source argument – resolved later by run_app
    return None

def _default_dest() -> Optional[Path]:
    # Default for the destination argument – resolved later by run_app
    return None

def _default_save_map() -> Optional[Path]:
    return None
""")

# Constant monitor_dir default
Updatable(code=r"""
monitor_dir = True
""")

# Arg_Defintions - Defines GUI / TUI / Argument behaviours using ArgSpec class
Updatable(code=r"""
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
""")

# GUI size + styling
Updatable(code=r"""
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
    "geometry": "800x400",
    "bg_color": "#84a9bf", #f9F9f9 Black - not recommended
    "fg_color": "#000000", #000000 White
    "font": ("Arial", 10),
}
""")

# ========================= Cross platform helpers ========================= #

# IO path handler - reject same source / dest
Updatable(code=r"""
def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except Exception:
        try:
            ar = a.resolve(strict=False)
            br = b.resolve(strict=False)
            return os.path.normcase(str(ar)) == os.path.normcase(str(br))
        except Exception:
            return os.path.normcase(str(a)) == os.path.normcase(str(b))
""")

# _is_windows _win_long - Prefix long paths on Windows to avoid MAX_PATH issues
Updatable(code=r"""
def _is_windows() -> bool:
    return os.name == "nt"

def _win_long(s: str) -> str:
    if not _is_windows():
        return s
    if s.startswith("\\\\?\\"):
        return s
    p = Path(s)
    try:
        abs_s = str(p.resolve())
    except Exception:
        abs_s = str(p.absolute())
    return "\\\\?\\" + abs_s
""")

# conserve metadata
Updatable(code=r"""
if sys.platform.startswith("win"):
    # ==== preserve creation time on windows when moving across drives
    # Try to prepare Windows creation-time setter at import
    try:
        import pywintypes
        import win32file, win32con

        def _set_windows_creation_time(path: str, ctime: float) -> None:
            handle = win32file.CreateFile(
                path,
                win32con.GENERIC_WRITE,
                win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
                None,
                win32con.OPEN_EXISTING,
                win32con.FILE_ATTRIBUTE_NORMAL,
                None,
            )
            win32file.SetFileTime(
                handle,
                pywintypes.Time(ctime),  # creation time
                None,                    # last access time
                None,                    # last write time
            )
            handle.close()

    except ImportError:
        import ctypes
        from ctypes import wintypes

        FILE_WRITE_ATTRIBUTES = 0x100
        OPEN_EXISTING = 3
        FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

        CreateFile = ctypes.windll.kernel32.CreateFileW
        SetFileTime = ctypes.windll.kernel32.SetFileTime
        CloseHandle = ctypes.windll.kernel32.CloseHandle

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        def _to_filetime(timestamp):
            # Convert Python timestamp to Windows FILETIME (100-ns intervals since Jan 1, 1601)
            ns = int(timestamp * 1e7) + 116444736000000000
            return FILETIME(ns & 0xFFFFFFFF, ns >> 32)

        def _set_windows_creation_time(path: str, ctime: float) -> None:
            handle = CreateFile(
                path, FILE_WRITE_ATTRIBUTES, 0, None, OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS, None
            )
            if handle == -1:
                raise ctypes.WinError()

            ft = _to_filetime(ctime)
            if not SetFileTime(handle, ctypes.byref(ft), None, None):
                raise ctypes.WinError()
            CloseHandle(handle)

else:
    # On non-Windows, no-op
    def _set_windows_creation_time(path: str, ctime: float) -> None:
        return

""")

# window focus manager
Updatable(code=r"""
def take_focus() -> None:
    try:
        if sys.platform.startswith("win"):
            import ctypes
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            hWnd = kernel32.GetConsoleWindow()
            if hWnd:
                user32.ShowWindow(hWnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hWnd)
    except Exception:
        pass
""")

# NormalisedStr normalise - Collate like terms via COLLATE_MAPPING
Updatable(code=r"""
class NormalizedStr(str):
    def normalize(self, mapping: dict):
        flat_map = {}
        for key, vals in mapping.items():
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                flat_map[v.lower()] = key

        # Escape and sort by length (longer phrases first)
        escaped = sorted(map(re.escape, flat_map.keys()), key=len, reverse=True)

        # Match only whole words or phrases using word boundaries
        # \b doesn't work well for multi-word phrases, so use lookarounds instead
        pattern = re.compile(
            r'(?<!\w)(' + '|'.join(escaped) + r')(?!\w)',
            flags=re.IGNORECASE
        )

        # Replacement preserving original case lookup
        def replacer(match):
            matched_text = match.group(0).lower()
            return flat_map.get(matched_text, match.group(0))

        return pattern.sub(replacer, self)
""")

# _robust_move handler
Updatable(code=r"""
def _robust_move(
    src: Path,
    dst: Path,
    retries: int = 3,
    backoff: float = 0.4,
    preserve_times: bool = True,
) -> None:
    src = Path(src)
    dst = Path(dst)

    if preserve_times:
        stat = src.stat()
        atime, mtime = stat.st_atime, stat.st_mtime
        ctime = getattr(stat, "st_ctime", None)
    else:
        atime = mtime = ctime = None

    src_s, dst_s = str(src), str(dst)

    for attempt in range(retries):
        try:
            shutil.move(src_s, dst_s)

            if preserve_times:
                os.utime(dst_s, (atime, mtime))
                if sys.platform.startswith("win") and ctime is not None:
                    _set_windows_creation_time(dst_s, ctime)

            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
""")

# same_filesystem - used to determine whether to nop metadata conservation
Updatable(code=r"""
def same_filesystem(a: Path, b: Path) -> bool:
    if sys.platform.startswith("win"):
        return os.path.splitdrive(str(a))[0].lower() == os.path.splitdrive(str(b))[0].lower()
    else:
        return a.stat().st_dev == b.stat().st_dev
""")

# suppress HF transformers warnings
Updatable(code=r"""
hf_logging.set_verbosity_error()
""")

# Size & device utilities
Updatable(code=r"""
def _gib_str(nbytes: int, floor_gib: int = 1) -> str:
    gib = max(floor_gib, nbytes // (1024 ** 3))
    return f"{gib}GiB"
def _gib_float(nbytes: int) -> float:
    return nbytes / float(1024**3)
""")

# Device selection - preferences cuda if avail
Updatable(code=r"""
def ensure_torch_with_cuda() -> torch.device:
    try:
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0

    cuda_available = bool(num_gpus) and torch.cuda.is_available()
    print(f"Detected {num_gpus} GPU(s).")

    if cuda_available:
        try:
            name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available. GPU: {name}")
        except Exception:
            print("✅ CUDA available.")
        return torch.device("cuda")

    if num_gpus > 0 and not torch.cuda.is_available():
        print("⚠️ GPU detected, but this PyTorch build has no CUDA. Proceeding on CPU.")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("✅ Using Apple Metal (MPS).")
        return torch.device("mps")

    print("❌ No compatible GPU found. Using CPU.")
    return torch.device("cpu")
""")

# cpu / gpu memory budgeting
Updatable(code=r"""
def query_memory(fraction: float = 0.85) -> dict:
    mem = {}
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info()
                mem[i] = int(free * fraction)
    except Exception:
        pass

    try:
        import psutil
        mem["cpu"] = int(psutil.virtual_memory().available * fraction)
    except Exception:
        mem["cpu"] = 2 * 1024**3  # fallback 2GiB
    return mem
""")

# construct max_memory dict for transformers.from_pretrained
Updatable(code=r"""
def build_max_memory(fraction: float = 0.85) -> dict:
    mem_bytes = query_memory(fraction)
    return {k: _gib_str(v) for k, v in mem_bytes.items()}
""")

# Caption text & path safety START_PHRASE MEDIA_PHRASE INVALID_PATH_CHARS
# clean_caption() slugify() truncate_by_bytes truncate_filename
Updatable(code=r"""
START_PHRASE = re.compile(r'^\s*(?:this\s+is\b|there\s+is\b)\s*', re.IGNORECASE)
MEDIA_PHRASE = re.compile(
    r'^\s*(?:a|an|close\s*-?\s*up)?\s*(?:close\s*-?\s*up\s+)?(?:image|illustration|screenshot|photograph|photo|picture|drawing|sketch|3d\srender|rendition|painting)?\s+of\s+',
    re.IGNORECASE
)
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]+')  # Windows filename compliance
# MULTISPACE = re.compile(r'\s+')

def clean_caption(caption: str) -> str:
    s = START_PHRASE.sub("", caption)
    s = MEDIA_PHRASE.sub("", s)
    return s.strip()

def slugify(seg: str) -> str:
    # Normalize to preserve Unicode meaning across platforms
    seg = unicodedata.normalize("NFKC", seg.strip().lower())
    # remove apostrophe
    seg = seg.replace("'", "")
    # Replace invalid path chars with dash; allow spaces
    seg = INVALID_PATH_CHARS.sub("-", seg)
    seg = seg.replace("/", "-")
    return seg or "misc"

def truncate_by_bytes(s: str, max_bytes: int, encoding="utf-8") -> str:
    encoded = s.encode(encoding)
    if len(encoded) <= max_bytes:
        return s
    truncated = encoded[:max_bytes]
    return truncated.decode(encoding, errors="ignore")

def truncate_filename(filename: str, max_bytes: int, encoding="utf-8") -> str:
    p = Path(filename)
    ms = str(datetime.now().microsecond // 1000)
    timestamp = "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ms
    stem, suffix = p.stem, p.suffix
    room = max_bytes - len(timestamp) - len(suffix.encode(encoding))
    truncated_stem = truncate_by_bytes(stem, room, encoding)
    return str((truncated_stem + "_" + timestamp))  # returns stem_with_timestamp (no ext)
""")

# monitor directory + relaunch
Updatable(code=r"""
def wait_for_new(src_dir: Path, args, images: Optional[List[Path]] = None, dry_run: bool = False):
    # Watch for new image files in the given directory and restart the script
    # with the current default arguments once any new files appear.
    # Parameters
    # ----------
    # src_dir : Path
    #    Directory to monitor for new images.
    # args : argparse.Namespace
    #    Current parsed argument namespace.
    # images : Optional[List[Path]]
    #    List of already processed images (may be None or empty).
    # dry_run : bool
    #    If True, disables watchdog behavior.

    # -------------------------------------------------------------------
    # Honour the persisted preferences when monitoring for new files.
    # -------------------------------------------------------------------

    saved_prefs: Dict[str, Any] = {}
    if prefs_file.is_file():
        try:
            with open(prefs_file, "r", encoding="utf-8") as pf:
                saved_prefs = json.load(pf) or {}
        except Exception:
            saved_prefs = {}

    # Evaluate callable defaults once
    computed_defaults: Dict[str, Any] = {}
    for spec in ARG_DEFINITIONS:
        val_def = spec.default
        if callable(val_def):
            try:
                val_def = val_def()
            except Exception:
                val_def = None
        computed_defaults[spec.name] = val_def

    # Merge saved preferences into args
    for spec in ARG_DEFINITIONS:
        name = spec.name
        if name in saved_prefs:
            pref_val = saved_prefs.get(name)
            current_val = getattr(args, name, None)
            default_val = computed_defaults.get(name)
            try:
                if current_val == default_val or current_val is None:
                    if pref_val is None:
                        setattr(args, name, None)
                    elif spec.type is Path:
                        setattr(args, name, Path(pref_val))
                    elif spec.type is bool:
                        setattr(args, name, bool(pref_val))
                    elif spec.type is int:
                        setattr(args, name, int(pref_val))
                    else:
                        setattr(args, name, pref_val)
            except Exception:
                pass

    # Determine whether monitoring is enabled
    if getattr(args, "dry_run", False):
        raise SystemExit(0)
    if not getattr(args, "monitor", False):
        raise SystemExit(0)

    # Track already-processed file names
    try:
        processed_names = {p.name for p in images} if images else set()
    except Exception:
        processed_names = set()

    print(f"Watching '{src_dir}' for new images…")

    while True:
        try:
            current_names = {p.name for p in list_images(src_dir)}
        except Exception:
            current_names = processed_names

        new_names = [n for n in (current_names - processed_names)
                     if (src_dir / n).suffix.lower() in IMAGE_EXTS]

        if new_names:
            print(f"Detected {len(new_names)} new image(s). Restarting to process them…")

            try:
                # ---------------------------------------------------------
                # Build proper restart command for both frozen and normal
                # ---------------------------------------------------------
                if getattr(sys, 'frozen', False):
                    # PyInstaller (OneFile or OneDir)
                    # In OneDir mode, sys._MEIPASS points to the temp extraction dir
                    # We only need the executable itself
                    executable_path = Path(sys.executable).resolve()
                    argv = [str(executable_path)]
                else:
                    # Standard Python script execution
                    python_path = Path(sys.executable).resolve()
                    script_path = Path(__file__).resolve()
                    argv = [str(python_path), str(script_path)]

                # Build remaining arguments from args namespace
                for spec in ARG_DEFINITIONS:
                    name = spec.name
                    val = getattr(args, name, None)

                    # Positional argument (e.g. "source")
                    if spec.cli_flags is None:
                        if val is not None:
                            argv.append(str(val))
                        continue

                    # Boolean flags
                    if getattr(spec, "action", "") == "store_true":
                        if val:
                            argv.append(spec.cli_flags[0])
                        continue

                    # Non-boolean args (only if not default)
                    default_val = computed_defaults.get(name)
                    if val is None or val == default_val:
                        continue
                    argv.extend([spec.cli_flags[0], str(val)])

                # Clean duplicates or empty entries
                argv = [a for a in argv if a]

                print(f"Restarting script with defaults: {' '.join(argv)}")

                # Execute the new process in-place
                os.execv(argv[0], argv)

            except Exception as e:
                print(f"Failed to restart automatically: {e}")
                break

        time.sleep(5.0)
""")

# BlipCaptioner - construct BLIP captioner
Updatable(code=r"""
class BlipCaptioner:
    def __init__(self, dtype=torch.float16, device_map="auto"):
        max_mem = build_max_memory(0.85)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large", use_fast=True
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            dtype=dtype,
            device_map=device_map,
            max_memory=max_mem,
        )
        self.model.eval()
        self.device: Optional[torch.device] = None  # torch.device set externally

    # Caption batch of PIL images
    @torch.inference_mode()
    def caption_batch(self, pil_images: List[Image.Image]) -> List[str]:
        imgs = [im.convert("RGB").resize((488, 488)) for im in pil_images]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_length=90,
            num_return_sequences=1,
            num_beams=12,
            num_beam_groups=2,
            do_sample=False,
            length_penalty=1.8,
            repetition_penalty=2.5,
            diversity_penalty=1.8,
            trust_remote_code=True,
        )
        return [self.processor.decode(seq, skip_special_tokens=True) for seq in out]
""")

# CONSTANTS - Default values. Update possible using config files
# Style Taxonomy used to generate CLIP prompts dataset
ROOT_CATEGORIES = [# zero shot catagory embeddings
    "astrophotography", "photography", "illustration", "fantasy art", "macro photography", "sketch", "anime", "painting", "abstract",
]

# zero shot classification subject embeddings
SUBJECTS = [
    "fairy", "fashion-model", "extreme weather", "reef", "lightning", "architecture", "cat", "dog",
    "bat", "dragon", "bird", "rat", "squirrel", "kangaroo", "pig", "crocodile", "koala", "monkey",
    "horse", "duck", "platypus",  "giraffe", "elephant", "hippopotomus", "zebra", "rhinocerous",
    "leopard", "octopus", "train-station", "space-station", "space", "landscape", "cityscape",
    "river", "car", "plane", "ship", "artifact", "interior-design", "animal", "flower", "waterfall",
    "sunset", "insect", "comet", "demon", "galaxy", "lighthouse", "wind turbine", "windmill",
    "dragon", "spider", "woman", "man", "black hole", "skateboarder", "surfer", "snowboarder",
    "bicycle", "cyclist", "motorbike", "volcano", "valley",
]

PROMPT_TEMPLATES: Dict[str, List[str]] = {# zero shot prompt templates constructed from category and subject
    "macro photography": [
        "a photo of a {subject} with a blurry background",
        "a clear and detailed photo of a {subject}",
        "macro-photography of a {subject}",
    ],
    "astrophotography": [
        "a photograph of a stars and nebula",
        "a photo of a galaxy and stars",
        "a realistic photo of a galaxy with stars",
    ],
    "photography": [
        "a professional portrait of a {subject}",
        "a photograph of a {subject}",
        "a photo of a {subject}",
        "a realistic photo of a {subject}",
        "a close-up photograph of a {subject}",
        "a documentary still of extreme weather",
        "a night landscape photo of a {subject}",
    ],
    "illustration": [
        "an illustration of a {subject}",
        "a detailed illustration of a {subject}",
        "a drawing of a {subject}",
    ],
    "abstract": [
        "abstract concept of a {subject}",
        "abstract art of a {subject}",
        "fractal image a {subject}",
    ],
    "painting": [
        "an impressionistic painting of a {subject}",
        "a watercolor painting of a {subject}",
        "a pointillism painting of a {subject}",
        "an abstract painting of a {subject}",
    ],
    "fantasy art": [
        "fantasy art of a {subject}",
        "a fantasy artwork of a {subject}",
        "concept art of a {subject}",
        "art of a human {subject} with wings",
    ],
    "sketch": [
        "a sketch of a {subject}",
        "a pencil sketch of a {subject}",
        "a rough sketch of a {subject}",
        "charcoal drawing of a {subject}",
    ],
    "anime": [
        "an anime‑style {subject}",
        "anime artwork of a {subject}",
        "an anime illustration of a {subject}",
    ],
}

# -----------------------------------------------------------------------------
# 'Strict' categorisation forcing. Used to push very specific terms
# into a particular category / category not defined in ROOT_CATEGORIES
# Filtering takes first match - overrides category if found in caption
# key position in dictionary determines precedence.
FILTER = {  # map specific overrides to category folder based on caption content
    "nude": "NSFW", "naked": "NSFW", "porn": "NSFW", "areola": "NSFW", "nipple": "NSFW", "nsfw": "NSFW", "phallus": "NSFW", "penis": "NSFW",
}

# -----------------------------------------------------------------------------
# Subject override heuristics
#
# BLIP captions can sometimes include words that more accurately describe the
# depicted subject than the CLIP model's initial prediction.  This
# dictionary allows subject names to be overridden when specific keywords
# appear in the generated caption.  The structure supports two forms of
# overrides:
#
#     SUBJECT_OVERRIDE_RULES = {
#         "original_subject": "keyword",
#         "other_subject": {
#             "new_subject": ["keyword1", "keyword2", ...],
#             ...
#         },
#         ...
#     }
#
# A simple string value indicates that when the caption contains the given
# keyword, the original subject should be replaced with the keyword itself.
# For nested dictionaries, each key represents a candidate new subject and
# the corresponding value is a sequence of keywords.  All keywords in the
# sequence must be present in the lowercase caption in order for the
# override to apply.  If no keywords match, the original subject is
# preserved.  Leaving this dictionary empty disables subject override
# heuristics.
# Subkeys intended as directory overrides may contain a nested structure:
# IE: 
pets = "cat", "dog"
wild_animals = (
    "tiger", "leopard", "girrafe", "crocodile", "gorilla", "snake", "eagle",
    "elephant", "zebra", "rhinocerous", "hippopotomus", "pig", "duck", "squirrel",
    "kangaroo", "platypus", "koala", "monkey", "horse"
)
SUBJECT_OVERRIDE_RULES: Dict[str, object] = {
    "fairy": "angel",
    "angel": ["woman","wings"],
    "lighthouse": {
        "seascape": "castle",
    },
    "windmills": "wind turbine",
    "demon": {
       "devil": ["red", "horn", "tail"],
       "minotaur": "minotaur",
    },
    "fashion-model": {
       "angel": ["woman", "wings"],
    },
}

# Dictionary structure for rules application is as Above
STRICT_CATEGORY_OVERRIDE_RULES: Dict[str, object] = {
    "astrophotography": {
        "landscape / Sunset": "sunset",
        "landscape / Sunrise": "sunrise",
        "landscape / Sun Rays": "sun rays",
        "fantasy art / angel": {("man","wings",), ("woman","wings",),},
        "painting / humanoid": {("painting","woman",),("painting","man",),("painting","person",),("painting","child",),("painting","human",),},
        "fantasy art / humanoid": {("woman"), ("human"), ("person"), ("child"), ("man")},
    },
    "astrophotography / fashion-model": {
        "photography / fashion-model": {("model",),("woman",),("man",)},
    },
    "fashion-model": {
        "fantasy art": ["woman","moon",],
    },
    "fantasy art": {
        "fantasy art / people": {("woman",),("man",),("human",),("person",),("child",)},
    },
    "photography": {
        "macro photography": "a close up",
    },
}

# this dictionary is used to collate similar subjects together for directory naming
# Target is the directory name as dictionary key, values are the like subjects to be collated
COLLATE_MAPPING = { 
    "sunset": ["sunrise", "sun rays",],
    "angels and fairies": ["angel", "fairy",],
    "humanoid": ["woman", "man", "humanoid",],
    "landscape": ["lightning", "windmills", "wind turbine", "lighthouse", "farm", "valley",],
}

# Ensure all ROOT_CATEGORIES keys exist
for root in ROOT_CATEGORIES:
    STRICT_CATEGORY_OVERRIDE_RULES.setdefault(root, {})

    for pet in pets:
        # Add pets under each root
        STRICT_CATEGORY_OVERRIDE_RULES[root][f"{root} / animal / pet / {pet.lower()}"] = pet.lower()
    # Add wild animals under each root
    for animal in wild_animals:
        STRICT_CATEGORY_OVERRIDE_RULES[root][f"{root} / animal / wild / {animal.lower()}"] = animal.lower()

# -----------------------------------------------------------------------------
# Category override heuristics
#
# Some images can plausibly belong to multiple categories.  The CLIP model
# returns similarity scores (stored in ``category_sums``) for each root
# category, and the highest score is selected as the best category.  However,
# captions may contain clues that suggest a different artistic medium (e.g.
# "sketch" or "anime"), and the category scores for these secondary
# categories might be only marginally lower than the best score.  To better
# reflect the true intent of the caption, this dictionary allows certain
# categories to be chosen when they are mentioned explicitly in the caption and
# the score difference relative to the best category is within a threshold.
#
# The structure is::
#
#     CATEGORY_OVERRIDE_RULES = {
#         "target_category": {
#             "keyword": score_threshold,
#             ...
#         },
#         ...
#     }
#
# When a ``keyword`` is found in the lowercase caption and the difference
# ``category_sums[best_category] - category_sums[target_category]`` is less
# than or equal to ``score_threshold``, the ``target_category`` will replace
# the best category.  You can adjust the keywords and thresholds to suit your
# dataset; leaving this dictionary empty disables the heuristic.
CATEGORY_OVERRIDE_RULES: Dict[str, Dict[str, float]] = {
    # Example rule: override to "sketch" if the caption mentions "sketch" or
    # "drawing" and the category score difference is small.  The threshold
    # values here are conservative; you may need to tune them empirically.
    # higher values = likelier to overide, 1 = forced overide
    # RECOMMENDATION : scale threshold values based on specifity of the target subkey of the caption.
    # IE:              "photo"            - a common occurance in captions: threshold <= 0.2
    #                  "man walking down" - very specific                 : threshold >= 0.98 
    #
    # "sketch": {"sketch": 0.2, "drawing": 0.2, "pencil": 0.3},
    # "anime": {"anime": 0.2, "manga": 0.3, "chibi": 0.3},
    # Add further categories and keywords below as needed.
    "photography": {
        "interior design": 1, "coffee table": 1, "man walking down": 0.98, "catwalk": 0.15,
        "photo": 0.1, "has taken a picture": 0.15, "wild-life": 0.15, "wildlife": 0.15, "photograph": 0.15,
        "photograph": 0.15, "portrait": 0.1, "documentary": 0.15, "landscape": 0.15,
        "make-up": 0.15, "make up": 0.15,
    },
    "anime": {"anime": 0.20, "manga": 0.25, "chibi": 0.25, "studio ghibli": 0.25},
    "sketch": {"black and white drawing": 0.2, "pencil sketch": 0.3, "crayon drawing": 0.15},
    "macro photography": {"flower": 0.15,"orchid": 0.15, "insect": 0.15, "butterfly": 0.15, "bee": 0.15, "damsel-fly": 0.15, "damselfly": 0.15, "dragon-fly": 0.15, "dragonfly": 0.15,"scorpian": 0.15, "spider": 0.15},
    "fantasy art": {"world of warcraft": 1, "dungeons and dragons": 1, "woman": 0.05,}
}

RULES_CONFIG = (
    "SUBJECTS",          # Subjects   that CLIP may select from. should be distinct and avoid conceptual Overlap.
    "ROOT_CATEGORIES",   # Catagories that CLIP may select from. should be distinct and avoid conceptual Overlap.
                         # - Categories defined in this dictionary must have corresponding keys in PROMPT_TEMPLATES
    "PROMPT_TEMPLATES",  # Templates used to generate CLIP embeddings for SUBJECTS ROOT_CATGORIES combinations
    "FILTER",            # Simple NSFW filtering based on the presence of defined words in the BLIP caption
    "COLLATE_MAPPING",   # Used to group conceptually-like SUBJECTS
    "STRICT_CATEGORY_OVERRIDE_RULES", # Used to preference Category based on content of the generated BLIP prompt. 
    "SUBJECT_OVERRIDE_RULES",         # Used to preference  Subject based on content of the generated BLIP prompt.
    "CATEGORY_OVERRIDE_RULES",        # Allows fine tuned biasing of category based on subject. 
)

# file extensions considered as images
IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# gathers image paths from target directory. path validated at args processing stage.
Updatable(code=r"""
def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
""")

# load_custom_rule_dicts() - allow rule configuration by user
Updatable(code=r"""
def load_custom_rule_dicts(*dict_names):
    # Dynamically loads one or more dictionaries from config/{name}.py if they exist.
    # Injects each loaded dictionary into the caller's global environment.
    # If loading fails for any reason, restores the previously defined global dictionary value.
 
    # Flatten tuple/list input
    if len(dict_names) == 1 and isinstance(dict_names[0], (list, tuple)):
        dict_names = dict_names[0]

    caller_globals = inspect.currentframe().f_back.f_globals

    for dict_name in dict_names:
        file_path = config_dir / f"{dict_name}.py"
        if not file_path.exists():
            continue  # skip silently

        # Fully dynamic import
        spec = importlib.util.spec_from_file_location(dict_name, file_path)
        module = importlib.util.module_from_spec(spec)
        loaded = True
        try:
            spec.loader.exec_module(module)
            new_dict = getattr(module, dict_name, None)
            # if not isinstance(new_dict, dict):
                # raise NameError(f"{dict_name} not defined as dict")
        except NameError:
            loaded = False
            print(f"⚠️ Custom rule {dict_name} in:\n   {file_path}\n   does not conform with formatting requirements.\n   Reverting to default.")
        except ValueError:
            loaded = False
            print(f"⚠️ Custom rule {dict_name} in:\n   {file_path}\n   Dictionary Values do not conform with formatting requirements.\n   Reverting to default.")
        except Exception as e:
            loaded = False
            print(f"⚠️ Failed to load custom rule from {file_path}\n: {e}\n Reverting to default.")
        
        if loaded:        
            caller_globals[dict_name] = new_dict
            print(f"✅ Custom rule applied for '{dict_name}'")

load_custom_rule_dicts(RULES_CONFIG)
""")

# OverrideString class
Updatable(code=r"""
class OverrideString(str):
    # A string subclass that knows about the caption context and can apply
    # override rules. Instances are immutable; :meth:`Apply_rules` returns a new
    # OverrideString or self if no rules match.
    # The ``OverrideString`` is a subclass of ``str`` that carries a caption context
    # and a flag indicating whether captioning is enabled.  It provides a method
    # :meth:`Apply_rules` to apply override heuristics based on a rules dictionary.
    # The rules dictionary must follow the same structure as
    # :data:`SUBJECT_OVERRIDE_RULES` or :data:`STRICT_CATEGORY_OVERRIDE_RULES`.
    # When invoked, the method will examine the stored caption (in lowercase) and
    # apply the first matching override for the current value.  If no override
    # matches or captioning is disabled, the original string instance is returned
    # unchanged. This utility allows category and subject overrides to be applied
    # in a uniform way via method calls on the current string value.

    def __new__(cls, value: str, cap_lower: str = "", no_captioning: bool = False):
        obj = super().__new__(cls, value)
        return obj

    def __init__(self, value: str, cap_lower: str = "", no_captioning: bool = False):
        self.cap_lower = cap_lower or ""
        self.no_captioning = no_captioning

    def _normalize_conditions(self, cond):
        # Normalize condition data structures so that:
        #  - single strings become single-element tuples
        #  - sets of strings become sets of single-element tuples
        #  - nested types are preserved

        if isinstance(cond, str):
            return (cond.lower(),)
        elif isinstance(cond, (list, tuple)):
            # Convert each element to lowercase string
            return tuple(str(c).lower() for c in cond)
        elif isinstance(cond, set):
            norm_set = set()
            for g in cond:
                if isinstance(g, str):
                    # convert bare string to tuple
                    norm_set.add((g.lower(),))
                elif isinstance(g, (list, tuple)):
                    norm_set.add(tuple(str(x).lower() for x in g))
                else:
                    # fallback: ignore invalid group types
                    continue
            return norm_set
        else:
            return cond  # leave as-is for unknown types

    def Apply_rules(self, override_rules: Dict[str, object]):
        # Apply override rules to this string based on the stored caption context.
        # Handles:
        #  - Simple string overrides
        #  - Dict of new_value -> keywords/list/tuple/set
        #  - Automatically normalizes malformed rule syntax

        if self.no_captioning:
            return self

        current_key = self.lower() if isinstance(self, str) else self
        rules = override_rules.get(current_key)
        if not rules:
            return self

        # Simple string override
        if isinstance(rules, str):
            keyword = rules.lower()
            if keyword and keyword in self.cap_lower:
                return OverrideString(rules, self.cap_lower, self.no_captioning)
            return self

        # Nested dictionary: new_value -> conditions
        elif isinstance(rules, dict):
            for new_value, cond in rules.items():
                cond = self._normalize_conditions(cond)

                # Single string or tuple: all items must appear
                if isinstance(cond, (tuple, list)):
                    if all(word in self.cap_lower for word in cond):
                        return OverrideString(new_value, self.cap_lower, self.no_captioning)

                # Set of groups (each group is tuple/list)
                elif isinstance(cond, set):
                    for group in cond:
                        group = self._normalize_conditions(group)
                        if isinstance(group, (tuple, list)):
                            if all(word in self.cap_lower for word in group):
                                return OverrideString(new_value, self.cap_lower, self.no_captioning)
                # Ignore invalid structures silently

            return self

        # Unknown rule type: return unchanged
        return self
""")

# load_clip()
Updatable(code=r"""
def load_clip(model_name: str = "ViT-B/32"):
    # Load a zero‑shot image–text model for classification.  If ``model_name``
    # refers to a Hugging Face model (detected by the presence of a slash or
    # absence from the built‑in CLIP model list), a compatible Transformers
    # model and processor are loaded.  Otherwise, an OpenAI CLIP model is
    # used via the ``clip`` library.  While loading Hugging Face models the
    # underlying libraries may emit repeated S3 download errors when network
    # connectivity is poor.  To avoid hanging indefinitely, this function
    # monitors logging output for a specific retry error message.  If more
    # than three occurrences are seen during model loading, a
    # ``ModelLoadError`` is raised so that the caller can prompt the user to
    # select a different model.

    # Parameters
    # ----------
    # model_name: str
    #    Name or identifier of the model to load.  Names without a forward
    #    slash are assumed to be OpenAI CLIP models.  Names with a slash
    #    (e.g. ``facebook/metaclip-2-worldwide-huge-quickgelu``) are treated
    #    as Hugging Face models.
    #
    # Returns
    # -------
    # tuple
    #    (model, preprocessor_or_processor, device, is_hf)
 
    os.environ["RUST_LOG"] = "error"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Determine whether to load a Hugging Face model or an OpenAI CLIP model.
    # The `clip` library exposes a list of built‑in model names via
    # ``available_models()``.  Any name not present in this list is
    # assumed to refer to a Hugging Face model.  This includes names
    # containing slashes (e.g. 'facebook/metaclip-2-worldwide-huge-quickgelu')
    # and any other identifier not in the default CLIP registry.  This
    # heuristic avoids erroneously attempting to load unknown models via
    # ``clip.load`` (which would raise a RuntimeError).
    try:
        available_clip_models = set(clip.available_models())
    except Exception:
        available_clip_models = {
            "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
            "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
        }
    # A model is considered a Hugging Face model if it is not in the
    # list of available CLIP models.  This captures names with slashes and
    # other custom identifiers.
    is_hf = model_name not in available_clip_models

    if is_hf:
        # Hugging Face path.  We wrap the download in a temporary S3 error watcher
        # so we can abort if too many retry errors are seen.  Note: do not
        # raise from inside logging; decisions are made after the calls.
        try:
            # Lazy import to avoid ImportError when transformers is not installed
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import transformers; required for Hugging Face models"
            ) from e
        # Start watching for S3 retry errors during the model and processor download
        with s3_error_watch(level=log.ERROR, threshold=3) as counter:
            # These calls may trigger logging of repeated S3 errors; our watch
            # will count them.
            try:
                model = AutoModel.from_pretrained(model_name)
                processor = AutoProcessor.from_pretrained(model_name)
            except RuntimeError as e:
                # specifically detect the CAS/xet-core error
                if "CAS service error" in str(e) or "Request failed after" in str(e):
                    raise ModelLoadError(
                        f"Download failed for {model_name}: {e}"
                    )
                else:
                    raise 
        # After the download attempt, examine the count and decide whether to abort
        if counter.count > counter.threshold:
            # Clean up partially constructed objects before aborting
            try:
                del processor
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            raise ModelLoadError(
                f"Aborting load of '{model_name}': observed {counter.count} S3 get_range failures."
            )
        # Attempt to handle any runtime errors that occurred outside logging
        # gracefully by letting them propagate normally.
        # When running on a GPU, decide whether to keep multi‑device support.  Standard
        # CLIP models (detected via `_is_standard_clip_model`) can leverage automatic
        # device maps; other models should be placed entirely on a single device.
        try:
            if device != "cpu" and torch.cuda.is_available() and not _is_standard_clip_model(model):
                model = model.to("cuda:0")
        except Exception:
            # If moving fails (e.g. due to model constraints) leave on CPU
            pass
        model.eval()
        return model, processor, device, True
    else:
        # OpenAI CLIP models are loaded locally via the clip library.  These
        # generally do not hit S3, so we load them directly without the watcher.
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        return model, preprocess, device, False
""")

# precompute Clip embedddings
Updatable(code=r"""
@torch.inference_mode()
def precompute_text(model, device, preprocess=None, is_hf: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
    # Precompute text embeddings for all category/subject prompt combinations.

    # This function generates prompts from ``ROOT_CATEGORIES`` and ``SUBJECTS``
    # using the templates in ``PROMPT_TEMPLATES``.  It then tokenises the
    # prompts and encodes them into feature vectors using either the OpenAI
    # CLIP model or a Hugging Face zero‑shot model.  The resulting vectors
    # are normalised and averaged per (category, subject) pair.

    # Parameters
    # ----------
    # model : torch.nn.Module
    #    The loaded model returned by ``load_clip``.
    # device : torch.device or str
    #    The device to perform computation on.
    # preprocess : callable or processor, optional
    #    For Hugging Face models this should be an instance of
    #    ``transformers.AutoProcessor``.  For OpenAI models it can be
    #    ``None`` (unused) or any value since tokenisation is handled via
    #    the ``clip`` library.
    # is_hf : bool, optional
    #    Flag indicating whether the model is a Hugging Face model.  When
    #    ``True``, the prompts are tokenised using the provided processor.

    # Returns
    # -------
    # dict
    #    A nested dictionary mapping category → subject → average embedding.

    feats: Dict[str, Dict[str, torch.Tensor]] = {}
    all_prompts: List[str] = []
    keys: List[Tuple[str, str]] = []
    for cat in ROOT_CATEGORIES:
        templates = PROMPT_TEMPLATES[cat]
        for sub in SUBJECTS:
            for tmpl in templates:
                all_prompts.append(tmpl.format(subject=sub))
                keys.append((cat, sub))
    # If using a Hugging Face model, ensure the model and inputs are on the same device.  Many
    # zero‑shot models loaded via AutoModel are initialised on CPU by default.  When a GPU is
    # available and ``device`` refers to CUDA, we move the model to the first CUDA device
    # explicitly.  Some models reject dicts for ``device_map``; therefore we avoid passing
    # device maps and instead call ``model.to(...)`` with a single device (e.g. "cuda" or
    # "cuda:0").  Without this call the subsequent token tensors would be moved to GPU while
    # the model remains on CPU, leading to ``Expected all tensors to be on the same device``
    # errors during embedding.
    if not is_hf:
        # Use OpenAI's CLIP tokeniser
        tokens = clip.tokenize(all_prompts).to(device)
        # model is already on the correct device from clip.load
        text_feats = model.encode_text(tokens)
    else:
        # Ensure model is on the target device when using Hugging Face models.  Only
        # apply this to non‑standard CLIP architectures; standard CLIP models retain
        # multi‑device support and should not be forced onto a single GPU.
        if not _is_standard_clip_model(model):
            try:
                # Convert strings like 'cuda' to explicit device indices where necessary
                if isinstance(device, str) and device != "cpu":
                    # Use CUDA device 0 explicitly when running on GPU to avoid device map issues
                    dev_str = "cuda:0" if device.startswith("cuda") else device
                    model = model.to(dev_str)
                elif isinstance(device, torch.device) and device.type != "cpu":
                    # Device is a torch.device; move to index 0 for cuda
                    dev = torch.device("cuda:0") if device.type == "cuda" else device
                    model = model.to(dev)
            except Exception:
                # If moving fails, continue with the model on its existing device
                pass
        # Use Hugging Face tokeniser via AutoProcessor to prepare text inputs
        if preprocess is None:
            raise ValueError("Hugging Face model requires a processor for tokenisation")
        # The processor may return additional keys; filter to keep only text inputs
        try:
            text_inputs = preprocess(text=all_prompts, return_tensors="pt", padding=True)
        except Exception:
            # Fallback: try using AutoTokenizer if available
            from transformers import AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(
                model.name_or_path if hasattr(model, "name_or_path") else preprocess.tokenizer.name_or_path  # type: ignore
            )
            text_inputs = tokenizer(all_prompts, return_tensors="pt", padding=True)
        # Determine the target device for tensors.  For non‑standard CLIP models,
        # the model and tensors are placed on a single GPU (if available).  Standard
        # CLIP models retain multi‑device support, so leave the tensors on CPU
        # and allow the transformers library to handle device placement.
        if not _is_standard_clip_model(model):
            if isinstance(device, str) and device != "cpu":
                target_dev = torch.device("cuda:0") if device.startswith("cuda") else torch.device(device)
            elif isinstance(device, torch.device) and device.type != "cpu":
                target_dev = torch.device("cuda:0") if device.type == "cuda" else device
            else:
                target_dev = torch.device("cpu")
            for k in list(text_inputs.keys()):
                if isinstance(text_inputs[k], torch.Tensor):
                    text_inputs[k] = text_inputs[k].to(target_dev)
                else:
                    text_inputs.pop(k)
        else:
            # For standard CLIP models, remove non‑tensor entries but do not move tensors
            for k in list(text_inputs.keys()):
                if not isinstance(text_inputs[k], torch.Tensor):
                    text_inputs.pop(k)
        # Compute text features using the Hugging Face API
        if hasattr(model, "get_text_features"):
            text_feats = model.get_text_features(**text_inputs)
        else:
            # Use forward pass and extract embeddings from outputs
            outputs = model(**text_inputs)
            # Attempt to use the text_embeds attribute if present
            if hasattr(outputs, "text_embeds"):
                text_feats = outputs.text_embeds
            else:
                # Fallback to first output (not ideal)
                text_feats = outputs[0]
    # Normalise embeddings
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    # Aggregate embeddings per (category, subject)
    aggregated: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    for key, vec in zip(keys, text_feats):
        aggregated.setdefault(key, []).append(vec)
    for (cat, sub), vecs in aggregated.items():
        stacked = torch.stack(vecs)
        avg = stacked.mean(dim=0)
        avg = avg / avg.norm(dim=-1, keepdim=True)
        feats.setdefault(cat, {})[sub] = avg
    return feats
""")

# In-memory image reuse
Updatable(code=r"""
class ImageBatch:
    # Holds PIL images and CLIP-preprocessed tensors for a list of paths.
    # Ensures each file is read from disk at most once.

    def __init__(self, paths: List[Path], preprocess, device: str):
        self.paths: List[Path] = []
        self.pils: List[Image.Image] = []
        self.clip_tensors: Optional[torch.Tensor] = None
        self._build(paths, preprocess, device)

    def _build(self, paths: List[Path], preprocess, device: str) -> None:
        ims_t: List[torch.Tensor] = []
        for p in paths:
            try:
                im = Image.open(p)
                # Fully load into memory to close file handles early
                im = im.convert("RGB").copy()
                im.load()
                self.paths.append(p)
                self.pils.append(im)
                # Only create CLIP tensors when a preprocess function is provided
                if preprocess is not None:
                    try:
                        ims_t.append(preprocess(im).unsqueeze(0))
                    except Exception:
                        # If preprocessing fails (e.g. due to mismatched API), skip tensor creation
                        pass
            except (UnidentifiedImageError, OSError):
                log.warning("Skipping unreadable image: %s", p)
        # For OpenAI CLIP models, build the tensor stack; otherwise leave None
        if preprocess is not None and ims_t:
            self.clip_tensors = torch.cat(ims_t, dim=0).to(device)
        elif preprocess is not None:
            # When no valid images were processed but preprocess exists, set an empty tensor
            self.clip_tensors = torch.empty(0, device=device)
        else:
            # Hugging Face models do not use clip_tensors
            self.clip_tensors = None
""")

# encode_clip_features()
Updatable(code=r"""
@torch.inference_mode()
def encode_clip_features(model, batch: ImageBatch, device, preprocess=None, is_hf: bool = False) -> torch.Tensor:
    # Encode image features for a batch of images.  For OpenAI CLIP models the
    # preprocessed tensors stored in the ``ImageBatch`` are used directly.
    # For Hugging Face models the images are processed on the fly using the
    # supplied processor.

    # Parameters
    # ----------
    # model : torch.nn.Module
    #    The loaded zero‑shot model.
    # batch : ImageBatch
    #    A batch of images and optional preprocessed tensors.
    # device : torch.device or str
    #    The device to perform computation on.
    # preprocess : callable or processor, optional
    #    The image processor for Hugging Face models.  Ignored for OpenAI
    #    models.
    # is_hf : bool, optional
    #    Flag indicating whether ``model`` is a Hugging Face model.

    # Returns
    # -------
    # torch.Tensor
    #    A tensor of shape ``(batch_size, feature_dim)`` containing L2‑normalised
    #    image embeddings.

    if not is_hf:
        # Use OpenAI CLIP image encoding
        if batch.clip_tensors is None or batch.clip_tensors.numel() == 0:
            return torch.empty(0, device=device)
        f = model.encode_image(batch.clip_tensors)
    else:
        # Hugging Face: process images on the fly via processor
        if preprocess is None:
            raise ValueError("Hugging Face model requires an image processor to encode images")
        # Prepare pixel values tensor using the provided processor.  Do not move
        # pixel values or the model for standard CLIP models; let transformers
        # handle device placement automatically.  For non‑standard CLIP models,
        # move everything to a single device (cuda:0) if a GPU is available.
        try:
            inputs = preprocess(images=batch.pils, return_tensors="pt")
        except Exception:
            # Fallback: try CLIPImageProcessor
            from transformers import CLIPImageProcessor  # type: ignore
            image_processor = CLIPImageProcessor.from_pretrained(model.name_or_path)
            inputs = image_processor(images=batch.pils, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            return torch.empty(0, device=device)

        if not _is_standard_clip_model(model):
            # Move model and inputs to a single device when not using a standard CLIP model
            try:
                if isinstance(device, str) and device != "cpu":
                    dev_str = "cuda:0" if device.startswith("cuda") else device
                    model = model.to(dev_str)
                    target_dev = torch.device(dev_str)
                elif isinstance(device, torch.device) and device.type != "cpu":
                    dev = torch.device("cuda:0") if device.type == "cuda" else device
                    model = model.to(dev)
                    target_dev = torch.device(dev)
                else:
                    target_dev = torch.device("cpu")
            except Exception:
                target_dev = torch.device("cpu")
            pixel_values = pixel_values.to(target_dev)
        else:
            # Standard CLIP model: do not move pixel values; use CPU placement
            target_dev = None  # placeholder; not used

        # Perform inference
        if hasattr(model, "get_image_features"):
            if not _is_standard_clip_model(model):
                f = model.get_image_features(pixel_values=pixel_values)
            else:
                f = model.get_image_features(pixel_values=pixel_values)
        else:
            # Use forward pass and extract image embeddings
            if not _is_standard_clip_model(model):
                outputs = model(pixel_values=pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)
            if hasattr(outputs, "image_embeds"):
                f = outputs.image_embeds
            else:
                f = outputs[0]
    # Normalise embeddings
    f = f / f.norm(dim=-1, keepdim=True)
    return f
""")

# Apply scoring
Updatable(code=r"""
def classify_from_features(
    image_feat: torch.Tensor,
    text_feats: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[str, str, Dict[str, float], Dict[str, float]]:
    subject_sums = {s: 0.0 for s in SUBJECTS}
    category_sums = {c: 0.0 for c in ROOT_CATEGORIES}
    for cat, submap in text_feats.items():
        for sub, tvec in submap.items():
            sim = float(image_feat @ tvec)
            subject_sums[sub] += sim
            category_sums[cat] += sim
    best_subject = max(subject_sums.items(), key=lambda kv: kv[1])[0]
    best_category = max(category_sums.items(), key=lambda kv: kv[1])[0]
    return best_category, best_subject, subject_sums, category_sums
""")

# filename handling: safe_segment()
Updatable(code=r"""
def safe_segment(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    return INVALID_PATH_CHARS.sub("_", name).strip()
""")

# filename handling: build_caption_filename()
Updatable(code=r"""
def build_caption_filename(caption: str, max_bytes: int, ext: str) -> str:
    cap = clean_caption(caption) or "image"
    cap = slugify(cap)
    stem_with_ts = truncate_filename(cap, max_bytes=max_bytes, encoding="utf-8")
    return f"{stem_with_ts}{ext}"
""")

# Adaptive BLIP batch sizing
Updatable(code=r"""
class AdaptiveBatchSizer:
    # Decides BLIP caption batch size using build_max_memory, dataset size, and backoffs on OOM.

    # Heuristics:
    #  - If CUDA: base on build_max_memory fraction of VRAM.
    #  - Else: base on build_max_memory fraction of system RAM.
    #  - Adjusts batch size downward if dataset is small or directory size is large.
    #  - BLIP_BATCH env var overrides.
    #  - On CUDA OOM, halves the batch and retries until 1.

    def __init__(self, image_dir: Path, default: int = 4, max_cap: int = 16):
        self.image_dir = image_dir
        self.default = max(1, default)
        self.max_cap = max_cap
        self._env_override = self._from_env()  # int | None

    def _from_env(self) -> Optional[int]:
        env = os.environ.get("BLIP_BATCH", "").strip()
        if env.isdigit():
            return max(1, int(env))
        return None

    def _dir_stats(self) -> Tuple[int, int]:
        # Return (num_images, total_bytes) in directory.
        total_bytes = 0
        num_images = 0
        for p in self.image_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                num_images += 1
                try:
                    total_bytes += p.stat().st_size
                except Exception:
                    pass
        return num_images, total_bytes

    def initial(self) -> int:
        if self._env_override is not None:
            return min(self._env_override, self.max_cap)

        mem = query_memory(0.90)  # fraction for batch sizing
        budget = mem.get(0) or mem.get("cpu", 2 * 1024**3)

        num_images, total_bytes = self._dir_stats()
        avg_size = total_bytes // max(1, num_images)
        est = budget // max(1, avg_size)
        log.info(
            "Dataset stats: %d images, total %.2f MB, avg %.1f KB/image. Budget: %.2f GiB. max --clip-batch at this avg=%d",
            num_images,
            total_bytes / (1024**2),
            avg_size / 1024 if avg_size else 0,
            _gib_float(budget),
            est,
        )

        return int(max(self.default, min(est, self.max_cap)))

    def backoff(self, current: int) -> int:
        return max(1, current // 2)
""")

# run_app() - executes main
Updatable(code=r"""
# ========================= Main ========================= #
# This function allows the script to be called both from the traditional command
# line and from an interactive prompt (see `main` below).
def run_app(args: argparse.Namespace) -> None:
    init = time.perf_counter()
    # Execute the image classification/caption/move pipeline using an argument
    # namespace.  This was extracted from the original `main` function so it
    # can be reused by both CLI and GUI invocation paths.

    # Parameters
    # ----------
    # args : argparse.Namespace
    #    Namespace containing the following attributes:
    #    - source (Path or None): path to directory of images.
    #    - dest (Path or None): destination root directory.
    #    - log (bool): enable logging of output paths.
    #    - clip_batch (int): batch size for CLIP image encoding.
    #    - dry_run (int/bool): if truthy, do not move files.
    #    - save_map (Path or None): optional path to write destinations JSON.

    # Resolve source and destination directories
    src_dir: Path = (args.source or Path.cwd()).resolve()
    dest_candidate: Path = (args.dest if args.dest is not None else (Path.cwd() / "images")).resolve()

    # Avoid moving into the same directory; fall back to `Processed_Images`
    if _same_path(dest_candidate, src_dir):
        dest_candidate = (Path.cwd() / "Processed_Images").resolve()

    dest: Path = dest_candidate

    if not src_dir.is_dir():
        raise SystemExit(f"Not a directory: {src_dir}")

    preserve_times = not same_filesystem(src_dir, dest)

    monitor_dir = bool(args.monitor)
    output_log = bool(args.log)
    dry_run = bool(args.dry_run)
    no_captioning = bool(args.no_captioning)
    # Ensure batch size is at least 1
    clip_batch = max(1, int(args.clip_batch))

    # Set up logging if requested
    log.basicConfig(level=log.INFO, format=" %(levelname)s | %(message)s ")

    # Load CLIP and BLIP models.  Allow the user to specify a custom CLIP
    # model via the --clip-model argument.  If not provided, the default
    # (ViT-B/32) is used.
    try:
        model_name = getattr(args, "clip_model", "ViT-B/32")
    except Exception:
        model_name = "ViT-B/32"
    # Load the zero‑shot model.  Hugging Face models return an extra
    # flag indicating that a Transformers processor should be used for both
    # text and image inputs.  OpenAI CLIP models provide a callable
    # preprocessor for images and rely on the ``clip`` library for text
    # tokenisation.  See ``load_clip`` for details on the return values.
    model, preproc_or_proc, device, is_hf = load_clip(model_name)

    # When using a Hugging Face model we need to pass the processor to
    # ``precompute_text`` so that prompts are tokenised correctly.  For
    # OpenAI CLIP models the processor is unused and can be ``None``.
    try:
        text_feats = precompute_text(
            model,
            device,
            preprocess=preproc_or_proc if is_hf else None,
            is_hf=is_hf,
        )
    except KeyError:
        print("A custom key defined in ROOT_CATEGORIES was missing in PROMPT_TEMPLATES")
        nul = input("press any key to exit.")
        raise SystemExit(1)
    except Exception as e:
        print(f"Unkown error - Failed to load model '{model_name}': {e}")
        print("Please check your network connection or try a different model.")
        nul = input("press any key to exit.")
        raise SystemExit(1)

    captioner = BlipCaptioner(dtype=torch.float16, device_map="auto")
    captioner.device = ensure_torch_with_cuda()
    log.info("Caption device: %s", captioner.device)

    images = list_images(src_dir)
    if not images:
        log.warning("No images found in %s", src_dir)
        try:
            wait_for_new(src_dir, args, images, dry_run)
        except KeyboardInterrupt:
            raise SystemExit(0)

    Destinations: Dict[str, str] = {}
    # A mapping from image paths (as strings) to the raw CLIP category scores
    # returned by ``classify_from_features``.  This dictionary is populated
    # when computing categories and subsequently consulted by the caption
    # heuristic to decide whether an alternate category should be chosen.
    CategoryScores: Dict[str, Dict[str, float]] = {}

    # Determine an initial caption batch size using heuristics
    bsizer = AdaptiveBatchSizer(image_dir=src_dir, default=4, max_cap=16)
    base_cap_bs = bsizer.initial()
    log.info("Initial BLIP caption batch size: %d", base_cap_bs)
    init_cost = time.perf_counter() - init
    start = time.perf_counter()

    # === Single pass pipeline over images in CLIP-sized batches ===
    for i in tqdm(range(0, len(images), clip_batch), desc="Classify > Caption > Move"):
        batch_paths = images[i:i + clip_batch]

        # Open each image ONCE and prepare CLIP tensors.  For Hugging
        # Face models we do not want to precompute CLIP tensors; pass
        # ``None`` to skip preprocessing in ImageBatch.  For OpenAI
        # models use the preprocessor returned from ``clip.load``.
        image_preprocess = preproc_or_proc if not is_hf else None
        mem_batch = ImageBatch(batch_paths, image_preprocess, device)
        if not mem_batch.paths:
            continue

        # CLIP features.  Provide the image processor when using
        # Hugging Face models to perform on‑the‑fly image processing.
        feats = encode_clip_features(
            model,
            mem_batch,
            device,
            preprocess=preproc_or_proc if is_hf else None,
            is_hf=is_hf,
        )

        # Compute categories/subjects for the in-memory batch.  In addition to
        # retrieving the best category/subject, we also capture the full
        # category score vector for each image.  This is stored in
        # ``CategoryScores`` for later use by the caption heuristic.
        for pth, feat_vec in zip(mem_batch.paths, feats):
            best_cat, best_sub, _, cat_sums = classify_from_features(feat_vec, text_feats)
            Destinations[str(pth)] = f"{best_cat} / {best_sub}"
            CategoryScores[str(pth)] = cat_sums

        # Caption + move using adaptive sub-batching over the in-memory PILs
        cap_bs = base_cap_bs
        start_idx = 0
        while start_idx < len(mem_batch.pils):
            end_idx = min(len(mem_batch.pils), start_idx + cap_bs)
            sub_pils = mem_batch.pils[start_idx:end_idx]
            sub_paths = mem_batch.paths[start_idx:end_idx]
            try:
                if no_captioning:
                    # When no_captioning is enabled we skip expensive BLIP caption
                    # generation and instead pass the filename forward via caps
                    # filenames.
                    caps = [p.stem for p in sub_paths]
                else:
                    caps = captioner.caption_batch(sub_pils)
            except RuntimeError as e:
                # Handle CUDA OOM by backing off and retrying
                msg = str(e).lower()
                if ("cuda" in msg or "out of memory" in msg or "cublas" in msg) and cap_bs > 1:
                    new_bs = bsizer.backoff(cap_bs)
                    log.warning("OOM at batch=%d — backing off to %d and retrying…", cap_bs, new_bs)
                    cap_bs = new_bs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue  # retry same window with smaller batch
                raise

            # Move files with generated captions
            for path, cap in zip(sub_paths, caps):
                pair = Destinations.get(str(path), "misc / misc")
                cat, sub = [seg.strip() for seg in pair.split("/", 1)]
                og_cat = cat
                og_sub = sub

                cap_lower = cap.lower()
                # ------- THRESHOLD CATEGORY OVERIDE HEURISTIC --------------
                # Heuristic: examine CLIP category scores to override the
                # classification if the caption suggests a different artistic
                # medium.  See ``CATEGORY_OVERRIDE_RULES`` for the keywords
                # and thresholds.  Only apply the override when the caption
                # mentions a keyword associated with an alternative category
                # and that category's score is within ``threshold`` of the
                # best category score.
                cat_scores = CategoryScores.get(str(path))
                if cat_scores:
                    best_category = cat

                    # ------------------------ LEGAL COMPLIANCE ----------------------------
                    cat_NSFW = False # Used to prevent  application of offensive phrases.
                    file_Nop = False # Used to prevent modification of likely illegal
                                     # files. Unit testing on file_Nop performed using
                                     # substitute terms for legal reasons.
                                         # compliance filtering requires captioning to be in use.
                                     # SPECIFIC COMPLIANCE OBJECTIVES:
                                     #  - not to propagate likely offensive captions.
                                     #    AI's are black boxes and it cannot be guarenteed
                                     #    that BLIP will not produce an inaccurate caption
                                     #    that results in offense, with potential for 
                                     #    inaccuracy vs the input image.
                                     #  - not to modify potentially illegal files to preserve
                                     #    chain of custody.

                    # Apply FILTER overrides to category if keyword found in caption
                    if not no_captioning:
                        for override_cat, kw_map in CATEGORY_OVERRIDE_RULES.items():
                            # Ignore cases where the override is the same as the current best
                            if override_cat == best_category:
                                continue
                            # Determine the score difference between the best and candidate categories
                            best_score = cat_scores.get(best_category, 0.0)
                            override_score = cat_scores.get(override_cat, 0.0)
                            score_diff = best_score - override_score
                            # Collect all matching keywords to consider the most permissive (highest) threshold.
                            matched_thresholds: List[float] = []
                            for kw, thresh in kw_map.items():
                                if kw in cap_lower:
                                    matched_thresholds.append(thresh)
                            if matched_thresholds:
                                # Choose the maximum threshold amongst matches.
                                max_thresh = max(matched_thresholds)
                                # Only override if the candidate category is close enough to the best
                                # according to the chosen threshold.
                                if score_diff <= max_thresh:
                                    cat = override_cat
                                    # Exit the outer loop after applying override
                                    break
                            # If a category override happened, exit both loops
                            if cat != best_category:
                                break

                if not no_captioning:
                    for key, value in FILTER.items():
                        if key in cap_lower:
                            cat = safe_segment(value)
                            if cat in ("NSFW","nsfw"):
                                cat_NSFW = True
                            break
                    if cat_NSFW: # legal compliance.
                        if "child" in cap_lower:
                            file_Nop = True
                            print(f"\033[31m!--! \033[33mcould not move: \033[0m{path.name}\033[31m !--!\033[0m")

                # if file_Nop == True reject further handling of potentially illegal content.
                if file_Nop:
                    continue
                else:
                    # ------------- SUBJECT AND CATEGORY CAPTIONING OVERRIDE HEURISTICS -------------
                    # Apply strict category and subject override rules via the OverrideString helper.
                    # When captioning is enabled, override the current category and subject by
                    # examining the stored caption for matching keywords defined in the rules.
                    if not no_captioning:
                        # Apply strict category overrides.  The current category is wrapped in an
                        # OverrideString along with the caption context so that the rules can
                        # inspect ``cap_lower`` internally.  The result is converted back to a
                        # plain string for subsequent processing.
                        cat_obj = OverrideString(cat, cap_lower, no_captioning).Apply_rules(STRICT_CATEGORY_OVERRIDE_RULES)
                        cat = str(cat_obj)
                        # Apply subject overrides.  Similarly, wrap the current subject and apply
                        # the subject override rules.  This returns either the original value or
                        # a new value depending on whether a rule matched.
                        sub_obj = OverrideString(sub, cap_lower, no_captioning).Apply_rules(SUBJECT_OVERRIDE_RULES)
                        sub = str(sub_obj)
                    
                    # Collapse like categories:
                    norm = NormalizedStr(cat)
                    cat = norm.normalize(COLLATE_MAPPING)
                    norm = NormalizedStr(sub)
                    sub = norm.normalize(COLLATE_MAPPING)

                    # Build sanitised cross platform directory path
                    cat_segments = [safe_segment(seg.strip()) for seg in cat.split('/') if seg.strip()]
                    cat_dir = Path(*cat_segments)  # Build nested directories safely
                    sub_segments = [safe_segment(seg.strip()) for seg in sub.split('/') if seg.strip()]
                    sub_dir = Path(*sub_segments)  # Build nested directories safely
                    
                    # reject duplicitous directory nesting
                    dir_target = str(sub_dir).lower()
                    dir_base = str(cat_dir).lower()
                    if dir_target in dir_base:
                        out_dir = dest / cat_dir
                    else:
                        out_dir = dest / cat_dir / sub_dir

                    if not no_captioning:
                        dir_obj = OverrideString(out_dir, cap_lower, no_captioning).Apply_rules(STRICT_CATEGORY_OVERRIDE_RULES)
                        out_dir = Path(dir_obj)

                    norm = NormalizedStr(out_dir)
                    out_dir = Path(norm.normalize(COLLATE_MAPPING))

                    # update Destinations mapping if category or subject changed
                    if cat != og_cat or sub != og_sub:
                        old_value = f"{og_cat} / {og_sub}"
                        new_value = f"{out_dir}"
                        key = str(path)
                        if Destinations.get(key) == old_value:
                            Destinations[key] = new_value
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        os.makedirs(_win_long(str(out_dir)), exist_ok=True)

                    ext = path.suffix
                    # If captioning is disabled, keep the original filename
                    # unchanged.  Otherwise, synthesise a descriptive filename
                    # based on the generated caption.  Original names are left
                    # intact to honour the user's "no captioning" preference.
                    if no_captioning:
                        dest_path = out_dir / path.name
                    else:
                        # legal compliance - reject application of likely offensive captions
                        if cat_NSFW:
                            cap = path.name
                        fname = build_caption_filename(cap, max_bytes=125, ext=ext)
                        dest_path = out_dir / fname

                    if dry_run:
                        log.info("[dry-run] %s  →  %s", path, dest_path)
                    else:
                        try:
                            _robust_move(path, dest_path, preserve_times=preserve_times)
                        except Exception:
                            ms = datetime.now().strftime("%f")[:3]
                            alt = out_dir / (dest_path.stem + f"_{ms}" + dest_path.suffix)
                            _robust_move(path, alt, preserve_times=preserve_times)

            start_idx = end_idx

        # Explicitly free tensors from this batch
        del mem_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # TODO relocate this mapping to occur after final
    # Optionally write destinations mapping to JSON
    if getattr(args, "save_map", None):
        out_path: Path = args.save_map
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(Destinations, f, indent=2, ensure_ascii=False)

    # Free model memory
    # Release resources associated with loaded models and processors.
    # ``preproc_or_proc`` may refer either to a callable preprocessor
    # (OpenAI CLIP) or a Transformers processor (Hugging Face).  Delete
    # it explicitly to free any underlying memory.
    del model, preproc_or_proc, device, text_feats, captioner

    # Optionally print destination mapping
    if output_log:
        print("\nDestinations mapping:")
        print(json.dumps(Destinations, indent=2, ensure_ascii=False))
    processing_elapsed = time.perf_counter() - start
    elapsed = processing_elapsed / len(images)
    print(f"Initialisation time: {init_cost:.3f} Processing Time: {processing_elapsed:.3f} ; sec/file: {elapsed:.3f} ; files: {len(images)}")

    
    # Watchdog mode: monitor the source directory for new images and re-exec
    # the script with the same arguments when new files appear.
    # Disable with WORDIFY_DISABLE_WATCHDOG=1 or when running --dry-run.
    try:
        wait_for_new(src_dir, args, images, dry_run)
    except KeyboardInterrupt:
        raise SystemExit(0)
""")

# prompt_for_args()
Updatable(code=r"""
def prompt_for_args() -> argparse.Namespace:
    # Launch an interactive configuration dialog and return an ``argparse.Namespace``
    # populated with argument values.  This function uses tkinter for a
    # user‑friendly GUI where available, and falls back to a simple text
    # interface (TUI) when tkinter cannot be imported (for example on
    # headless servers).  The behaviour and appearance of the GUI is driven
    # entirely by the entries in ``ARG_DEFINITIONS`` and ``GUI_SETTINGS``;
    # therefore adding a new argument only requires updating those data
    # structures.

    # Returns
    # -------
    # argparse.Namespace
    #    A namespace with attributes matching the ``name`` fields in
    #    ``ARG_DEFINITIONS``.

    # Compute initial defaults for interactive prompts.  Call callables to
    # determine dynamic defaults (e.g. for source/dest when unspecified).
    # ------------------------------------------------------------------
    # Persistent settings
    #
    # Load previously saved preferences from the user's home directory.  A
    # simple JSON file stores the last used values for each argument, and
    # these values override the built‑in defaults on subsequent runs.  This
    # mechanism ensures that user choices in the GUI are remembered across
    # sessions without relying on any GUI state.  If the file is missing
    # or unreadable the built‑in defaults are used.

    saved_prefs: Dict[str, Any] = {}
    if prefs_file.is_file():
        try:
            with open(prefs_file, "r", encoding="utf-8") as pf:
                saved_prefs = json.load(pf) or {}
        except Exception:
            saved_prefs = {}
    # Compute defaults by evaluating callables where necessary
    defaults: Dict[str, Any] = {}
    for spec in ARG_DEFINITIONS:
        val = spec.default
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        defaults[spec.name] = val
    # Override defaults with saved preferences, converting types as needed
    for spec in ARG_DEFINITIONS:
        if spec.name in saved_prefs:
            pref_val = saved_prefs.get(spec.name)
            try:
                if pref_val is None:
                    defaults[spec.name] = None
                elif spec.type is Path:
                    defaults[spec.name] = Path(pref_val)
                elif spec.type is bool:
                    defaults[spec.name] = bool(pref_val)
                elif spec.type is int:
                    defaults[spec.name] = int(pref_val)
                else:
                    defaults[spec.name] = pref_val
            except Exception:
                # On any conversion error, fall back to computed default
                pass

    # Attempt to build a GUI using tkinter.  If this fails e.g. running in
    # a non‑GUI environment we fall back to TUI.
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk, messagebox

        root = tk.Tk()
        # Apply window title
        root.title(GUI_SETTINGS.get("title", "Wordify - Image Sorter Setup"))
        # Apply geometry (size and optional position)
        geometry_str = GUI_SETTINGS.get("geometry")
        if geometry_str:
            root.geometry(geometry_str)
        # Prevent resizing by default to maintain layout integrity
        root.resizable(False, False)

        # Style configuration for ttk widgets
        bg_color = GUI_SETTINGS.get("bg_color") or "white"
        fg_color = GUI_SETTINGS.get("fg_color") or "black"
        font = GUI_SETTINGS.get("font")

        root.configure(background=bg_color)

        style = ttk.Style()
        style.configure("TLabel", font=font, foreground=fg_color, background=bg_color)
        style.configure("TEntry", font=font, foreground=fg_color)        # no bg override
        # style.configure("TButton", font=font)                          # respect theme
        style.configure("Custom.TButton",
                background=bg_color,
                foreground=fg_color,
                font=font
        )
        style.map("Custom.TButton", background=[("active", bg_color), ("pressed", bg_color)])

        # Map from ArgSpec to tkinter Variable
        var_map: Dict[str, Any] = {}

        # Helper to create browse functions for directories and files
        def _make_browse_dir(var):
            def func():
                initial = var.get() or os.getcwd()
                path = filedialog.askdirectory(title="Select directory", initialdir=initial)
                if path:
                    var.set(path)
            return func

        def _make_browse_file(var):
            def func():
                initial = var.get() or os.getcwd()
                path = filedialog.asksaveasfilename(
                    title="Select file", initialdir=initial, defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                )
                if path:
                    var.set(path)
            return func

        # Build main frame with padding
        frm = tk.Frame(root, bg=bg_color, padx=10, pady=10)
        frm.grid(row=0, column=0, sticky="nsew")

        # Layout row counter
        row_idx = 0
        # Create widgets for each argument
        for spec in ARG_DEFINITIONS:
            if not spec.gui:
                continue

            label_text = spec.gui_label or spec.name
            def_val = defaults.get(spec.name)

            # Convert default to string for entry fields
            if isinstance(def_val, Path):
                def_str = str(def_val) if def_val is not None else ""
            elif def_val is None:
                if spec.name == "source":
                    def_str = str(Path.cwd())
                elif spec.name == "dest":
                    def_str = str((Path.cwd() / "images").resolve())
                else:
                    def_str = ""
            else:
                def_str = str(def_val)

            if spec.gui_widget == "checkbox":
                # Checkboxes keep inline label text
                var = tk.BooleanVar(value=bool(def_val))
                chk = tk.Checkbutton(
                    frm,
                    variable=var,
                    text=label_text,
                    bg=bg_color,
                    fg=fg_color,
                    selectcolor=bg_color,
                    font=font,
                )
                chk.grid(row=row_idx, column=0, columnspan=3, padx=5, pady=3, sticky="w")
                var_map[spec.name] = var

            elif spec.gui_widget in {"dir", "savefile"}:
                # Descriptor label before entry
                lbl = tk.Label(
                    frm,
                    text=f"{label_text}:",
                    bg=bg_color,
                    fg=fg_color,
                    font=font,
                )
                lbl.grid(row=row_idx, column=0, padx=5, pady=3, sticky="w")

                var = tk.StringVar(value=def_str)
                entry = ttk.Entry(frm, textvariable=var, width=90)
                entry.grid(row=row_idx, column=1, padx=5, pady=3, sticky="ew")

                if spec.gui_widget == "dir":
                    browse = ttk.Button(frm, text="Browse…", command=_make_browse_dir(var))
                else:  # savefile
                    browse = ttk.Button(frm, text="Browse…", command=_make_browse_file(var))
                
                browse.grid(row=row_idx, column=2, padx=5, pady=3, sticky="e")
                var_map[spec.name] = var

            elif spec.gui_widget == "dropdown":
                # Dropdown combobox allows selection from a predefined list, but
                # remains editable so users can type arbitrary values.  A
                # descriptor label precedes the widget.
                lbl = tk.Label(
                    frm,
                    text=f"{label_text}:",
                    bg=bg_color,
                    fg=fg_color,
                    font=font,
                )
                lbl.grid(row=row_idx, column=0, padx=5, pady=3, sticky="w")
                # Use a StringVar to hold the selected/entered value
                var = tk.StringVar(value=def_str)
                # Determine available choices; fall back to empty tuple if None
                options = tuple(spec.choices) if spec.choices is not None else ()
                combo = ttk.Combobox(
                    frm,
                    textvariable=var,
                    values=options,
                    width=20,
                )
                # Leave state as default (editable) to allow custom values
                combo.grid(row=row_idx, column=1, padx=5, pady=3, sticky="ew")
                var_map[spec.name] = var

            else:
                # Plain entry with descriptor label
                lbl = tk.Label(
                    frm,
                    text=f"{label_text}:",
                    bg=bg_color,
                    fg=fg_color,
                    font=font,
                )
                lbl.grid(row=row_idx, column=0, padx=5, pady=3, sticky="w")

                var = tk.StringVar(value=def_str)
                entry = ttk.Entry(frm, textvariable=var, width=30)
                entry.grid(row=row_idx, column=1, padx=5, pady=3, sticky="ew")

                var_map[spec.name] = var
            row_idx += 1

        # After processing all arguments, insert a warning about CLIP model download
        # before the buttons (if the clip_model argument is present and shown in the GUI).
        # This ensures the warning label appears above the Start/Cancel buttons.
        if any(spec2.name == "clip_model" and spec2.gui for spec2 in ARG_DEFINITIONS):
            warn_lbl = tk.Label(
                frm,
                text="⚠ Clip classification model will be downloaded when first used. Model sizes and performance vary. Size != Accuracy",
                bg=bg_color,
                fg=fg_color,
                font=font,
            )
            warn_lbl.grid(row=row_idx, column=0, columnspan=3, padx=5, pady=3, sticky="w")
            row_idx += 1

        # Buttons
        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=row_idx, column=0, columnspan=3, pady=(10, 0))
        result: Dict[str, Any] = {}

        def on_ok():
            # Validate and collect values from the GUI.
            nonlocal result
            # Iterate through all specs and convert values
            for spec in ARG_DEFINITIONS:
                # For arguments not represented in the GUI, use the
                # previously determined default.  This ensures that
                # all preferences are persisted even if additional
                # non‑interactive options are added in the future.
                if not spec.gui:
                    val = defaults.get(spec.name)
                    result[spec.name] = val
                    continue
                raw = var_map[spec.name]
                # Extract and validate GUI values based on type
                if isinstance(raw, tk.BooleanVar):
                    val = bool(raw.get())
                else:
                    s = raw.get().strip()
                    if spec.type is int:
                        if not s:
                            val = defaults.get(spec.name)
                        else:
                            try:
                                val = int(s)
                                if val < 1:
                                    raise ValueError
                            except Exception:
                                messagebox.showerror("Invalid input", f"{spec.gui_label} must be a positive integer.")
                                return
                    elif spec.type is bool:
                        val = s.lower() in {"y", "yes", "true", "1"}
                    elif spec.type is Path:
                        if s == "":
                            val = None
                        else:
                            val = Path(s).expanduser()
                    else:
                        val = s or defaults.get(spec.name)
                result[spec.name] = val
            # Persist user selections to the preferences file.  Paths
            # are stored as strings for JSON serialisation and None is
            # represented explicitly.  This operation is best‑effort; a
            # failure to write should not interrupt the workflow.
            try:
                prefs_out: Dict[str, Any] = {}
                for spec in ARG_DEFINITIONS:
                    val = result.get(spec.name)
                    if isinstance(val, Path):
                        prefs_out[spec.name] = str(val) if val is not None else None
                    else:
                        prefs_out[spec.name] = val if val is not None else None
                with open(prefs_file, "w", encoding="utf-8") as pf:
                    json.dump(prefs_out, pf, ensure_ascii=False, indent=2)
            except Exception:
                # Ignore persistence errors silently
                pass
            # All values validated and preferences saved; close window
            root.destroy()
            # take_focus()

        def on_cancel():
            root.destroy()
            # take_focus()
            raise SystemExit(0)

        start_btn = ttk.Button(btn_frm, text="Start", command=on_ok, style="Custom.TButton")
        cancel_btn = ttk.Button(btn_frm, text="Cancel", command=on_cancel, style="Custom.TButton")
        start_btn.grid(row=0, column=0, padx=0)
        cancel_btn.grid(row=0, column=1, padx=0)
        # Bring window to front, cross-platform friendly
        root.lift()
        root.attributes("-topmost", True)
        root.after(10, lambda: root.attributes("-topmost", False))
        root.focus_force()
        root.mainloop()

        # Construct the namespace from collected results
        namespace_args: Dict[str, Any] = {}
        for spec in ARG_DEFINITIONS:
            val = result.get(spec.name, defaults.get(spec.name))
            # If a value is still callable (never invoked), evaluate it
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = None
            namespace_args[spec.name] = val
        return argparse.Namespace(**namespace_args)

    except Exception:
        # Fall back to a simple text‑based prompt if tkinter is unavailable
        print("\nInteractive mode: please enter the following values (press Enter to accept defaults).\n")
        collected: Dict[str, Any] = {}
        try:
            for spec in ARG_DEFINITIONS:
                if not spec.gui:
                    collected[spec.name] = defaults.get(spec.name)
                    continue
                # Determine default string for prompt
                def_val = defaults.get(spec.name)
                if isinstance(def_val, Path):
                    def_str = str(def_val) if def_val is not None else ""
                elif def_val is None:
                    if spec.name == "source":
                        def_str = str(Path.cwd())
                    elif spec.name == "dest":
                        def_str = str((Path.cwd() / "images").resolve())
                    else:
                        def_str = ""
                else:
                    def_str = str(def_val)
                prompt = spec.interactive_prompt or spec.gui_label or spec.name
                # Compose full prompt with default in brackets, except for boolean checkboxes which use (y/N)
                if spec.type is bool:
                    default_prompt = "y/N" if not def_val else "Y/n"
                    user_input = input(f"{prompt} ({default_prompt}): ").strip()
                    if user_input == "":
                        val = bool(def_val)
                    else:
                        val = user_input.lower() in {"y", "yes", "true", "1"}
                elif spec.type is int:
                    user_input = input(f"{prompt} [{def_str}]: ").strip()
                    if user_input == "":
                        val = def_val
                    else:
                        try:
                            iv = int(user_input)
                            if iv < 1:
                                raise ValueError
                            val = iv
                        except Exception:
                            print(f"Invalid value for {spec.name}; using default {def_str}.")
                            val = def_val
                elif spec.type is Path:
                    user_input = input(f"{prompt} [{def_str}] (leave blank for default): ").strip()
                    if user_input == "":
                        val = None if def_val is None else Path(def_val)
                    else:
                        val = Path(user_input).expanduser()
                else:
                    user_input = input(f"{prompt} [{def_str}]: ").strip()
                    val = user_input if user_input != "" else def_val
                collected[spec.name] = val
        except KeyboardInterrupt:
            raise SystemExit(0)
        # Persist collected values to preferences file.  We mirror the
        # behaviour of the GUI by writing the collected values to disk so
        # that subsequent runs can present them as defaults.  This is
        # best‑effort; errors during writing are ignored.
        try:
            prefs_out: Dict[str, Any] = {}
            for spec in ARG_DEFINITIONS:
                val = collected.get(spec.name)
                if isinstance(val, Path):
                    prefs_out[spec.name] = str(val) if val is not None else None
                else:
                    prefs_out[spec.name] = val if val is not None else None
            with open(prefs_file, "w", encoding="utf-8") as pf:
                json.dump(prefs_out, pf, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # Build namespace
        namespace_args: Dict[str, Any] = {}
        for spec in ARG_DEFINITIONS:
            val = collected.get(spec.name, defaults.get(spec.name))
            if callable(val):
                try:
                    val = val()
                except Exception:
                    val = None
            namespace_args[spec.name] = val
        return argparse.Namespace(**namespace_args)
""")

# main 
Updatable(code=r"""
def main() -> None:
    # Entry point for the script.  When invoked without command-line arguments
    # (i.e. only the script name is provided), an interactive prompt is shown
    # allowing the user to configure options such as source/destination
    # directories and batch sizes.  Otherwise, the traditional command-line
    # interface is used.

    # Use the interactive prompt if no additional command‑line arguments were
    # supplied.  Otherwise, parse the arguments using the unified definition
    # table built by ``build_parser``.  This makes adding or removing
    # arguments straightforward as all definitions live in ``ARG_DEFINITIONS``.
    if len(sys.argv) == 1:
        # Interactive invocation: present the GUI and handle model download
        # errors by re‑prompting the user to choose a different model.  We
        # loop until run_app completes successfully or the user cancels.
        while True:
            args = prompt_for_args()
            init = time.perf_counter()
            try:
                run_app(args)
            except ModelLoadError:
                # Inform the user that the chosen model failed to download.
                # When tkinter is available we display a messagebox; otherwise
                # we print to the console.  After the message we loop back
                # around to show the form again.
                # fall back to plain http download if xet-core fails
                # (e.g. due to firewall issues) 
                os.environ["HF_HUB_DISABLE_XET"] = "1"
                try:
                    import tkinter as tk
                    from tkinter import messagebox
                    # Create a transient root for the error dialog
                    err_root = tk.Tk()
                    err_root.withdraw()
                    messagebox.showerror(
                        "Model download failed",
                        "The selected CLIP model could not be downloaded due to repeated network errors.\n"
                        "Choose a different model or Try again."
                    )
                    err_root.destroy()
                except Exception:
                    # Fall back to a simple console message
                    print(
                        "Error: The selected CLIP model could not be downloaded due to repeated network errors.\n"
                        "Please choose a different model or Try again."
                    )
                continue
            # run_app completed without raising ModelLoadError; exit loop
            return

    # Build an ArgumentParser dynamically from ARG_DEFINITIONS
    ap = argparse.ArgumentParser(description="Batch CLIP classify → BLIP caption → move images")
    for spec in ARG_DEFINITIONS:
        # Skip arguments intended only for GUI
        if spec.cli_flags is None:
            # Positional argument: name acts as flag
            flags = [spec.name]
        else:
            flags = list(spec.cli_flags)
        kwargs: Dict[str, Any] = {"help": spec.help}
        # Use nargs for optional positional argument
        if spec.nargs:
            kwargs["nargs"] = spec.nargs
        # Determine parser action/type
        if spec.action:
            kwargs["action"] = spec.action
            # For store_true actions argparse will handle boolean conversion
            # so we don't need to provide type
        else:
            kwargs["type"] = spec.type
            # Evaluate default if callable
            if callable(spec.default):
                try:
                    default_val = spec.default()
                except Exception:
                    default_val = None
            else:
                default_val = spec.default
            # argparse expects default for optional arguments only; skip for
            # positional to allow run_app to resolve correctly
            if spec.cli_flags is not None:
                kwargs["default"] = default_val
        ap.add_argument(*flags, **kwargs)
    args = ap.parse_args()
    # For booleans parsed by store_true actions, argparse sets False by
    # default when the flag is omitted.  This matches our ArgSpec defaults.
    try:
        run_app(args)
    except ModelLoadError as e:
        # In non-interactive CLI mode we surface the error and exit with
        # non‑zero status.  The error message provides context about the
        # repeated download failures.
        log.error("%s", e)
        sys.exit(1)
""")

# launch main
Updatable(code=r"""
if __name__ == "__main__":
    main()
""")