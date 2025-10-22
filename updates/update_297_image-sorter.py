# github self updater. Opt-In.  
  #  Update file
  # v.0.0.2: When updating enabled, file is downloaded if missing locally regardless of local manifest metadata (updates.json).
  #          This change creates persistence for update files logged in the manifest while self-updating is enabled.
  # v.0.0.1: Modified to make updating opt-in via prefs file / gui checkbox.
__key__ = "4275339c153dab609e6e372805a08ffd6ed8813c97f01ea263b784760711dcc5"
__version__ = "0.0.2"

def fetch_all_updates_from_github(
    repo_owner: str,
    repo_name: str,
    branch: str = "main",
    subdir: str = "updates",
    config_dir: str = "./config",
    updates_dir: str = "./updates",
):
    # Fetch and cache self-update scripts from GitHub.

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
            pass  # ignore bad date and proceed

    # Build remote URLs for the manifest and raw files
    raw_base = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{subdir}"
    manifest_url = f"{raw_base}/update_ver.json"

    fetched_any = False
    try:
        # Download and parse the remote version manifest
        resp = requests.get(manifest_url, timeout=10)
        resp.raise_for_status()
        manifest = resp.json()  # Expecting dict of {filename: "ver:X.Y.Z key:..."}

        # Iterate through each declared update file
        for filename, remote_meta in manifest.items():
            # Only consider Python update scripts
            if not isinstance(filename, str) or not filename.endswith(".py"):
                continue

            remote_version, remote_key = _parse_version_key(remote_meta)
            local_meta = local_versions.get(filename, "")
            local_version = ""
            if isinstance(local_meta, str):
                local_version, _ = _parse_version_key(local_meta)

            local_path = os.path.join(updates_dir, filename)

            # If file is missing locally, download it regardless of version
            if not os.path.exists(local_path):
                print(f"[Info] '{filename}' listed in manifest but missing locally. Downloading...")
                file_url = f"{raw_base}/{filename}"
                try:
                    content_resp = requests.get(file_url, timeout=10)
                    content_resp.raise_for_status()
                    content = content_resp.text
                    remote_version = _extract_version_from_file(content)
                except requests.RequestException:
                    print(f"[Update Check] Failed to download missing file {filename}")
                    continue

                with open(local_path, "w", encoding="utf-8") as out:
                    out.write(content)

                local_versions[filename] = remote_meta
                print(f"[Fetched Missing] {filename} (ver {remote_version})")
                fetched_any = True
                continue  # skip version based update check when file missing locally.

            # Determine whether to update
            if filename not in local_versions or _is_version_newer(remote_version, local_version):
                if local_version == remote_version:
                    continue  # Skip if versions are identical

                # Download the actual update file
                file_url = f"{raw_base}/{filename}"
                try:
                    content_resp = requests.get(file_url, timeout=10)
                    content_resp.raise_for_status()
                    content = content_resp.text
                    remote_version = _extract_version_from_file(content)
                except requests.RequestException:
                    print(f"[Update Check] Failed to download {filename}")
                    continue

                with open(local_path, "w", encoding="utf-8") as out:
                    out.write(content)

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
