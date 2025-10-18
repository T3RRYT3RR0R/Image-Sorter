# GitHub self-updater  
  #  Update file
# key: 141a48a7e9a30852ab0bd6dbe4fd39db7be6a01027aa828a7f2fc9e41594c6a4

def fetch_all_updates_from_github(
    repo_owner: str,
    repo_name: str,
    branch: str = "main",
    subdir: str = "updates",
    config_dir: str = "./config",     # provide your config_dir path
    updates_dir: str = "./updates",   # provide your updates_dir path
):
    # Fetches all .py update files from a GitHub repo subdir and saves them locally.
    # Maintains a JSON metadata file mapping filename -> SHA256 hash.
    # Updates the hash when file content changes.
    # Only runs once per day. Timestamp stored in config_dir/update_check.txt.

    # Ensure directories exist
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(updates_dir, exist_ok=True)

    # Paths
    last_check_file = os.path.join(config_dir, "update_check.txt")
    updates_json = os.path.join(config_dir, "updates.json")

    # Load existing hash data
    if os.path.exists(updates_json):
        with open(updates_json, "r", encoding="utf-8") as f:
            try:
                update_hashes = json.load(f)
            except json.JSONDecodeError:
                update_hashes = {}
    else:
        update_hashes = {}

    # Skip if already checked today
    if os.path.exists(last_check_file):
        with open(last_check_file, "r", encoding="utf-8") as f:
            last_check_str = f.read().strip()
            try:
                last_check = dt.datetime.strptime(last_check_str, "%Y-%m-%d").date()
                if last_check == dt.date.today():
                    return  # already checked today
            except ValueError:
                pass

    # GitHub API endpoints
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{subdir}?ref={branch}"
    raw_base = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{subdir}"

    try:
        resp = requests.get(api_url, timeout=5)
        resp.raise_for_status()
        files = resp.json()

        fetched_any = False

        for f in files:
            name = f.get("name", "")
            if name.endswith(".py") and name.startswith("update_"):
                raw_url = f"{raw_base}/{name}"
                local_path = os.path.join(updates_dir, name)

                # Download file content
                content = requests.get(raw_url, timeout=5).text
                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                # Check if file is new or changed
                if update_hashes.get(name) != file_hash:
                    # Save new/updated file
                    with open(local_path, "w", encoding="utf-8") as out:
                        out.write(content)

                    # Update metadata
                    update_hashes[name] = file_hash
                    print(f"[Fetched] {name}")
                    fetched_any = True

        # Save updated metadata only if changes were made
        if fetched_any:
            with open(updates_json, "w", encoding="utf-8") as f:
                json.dump(update_hashes, f, indent=2)

        # Update timestamp
        with open(last_check_file, "w", encoding="utf-8") as f:
            f.write(dt.date.today().isoformat())

    except requests.RequestException as e:
        print(f"[Startup Update Check] Failed to fetch updates: {e}")
#‍ ∆eof
