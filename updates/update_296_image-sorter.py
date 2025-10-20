# github self updater  
  #  Update file
__key__ = "cbaaf0ba444cb9ac08c4a259cb5776426dc548b7a803dfcb82c515d5dd347e34"
__version__ = "0.0.1"

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
#‍ ∆eof
