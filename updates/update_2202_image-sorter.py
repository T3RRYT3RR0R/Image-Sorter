# run_app() - executes main  
  #  Update file
  # v.0.0.1 - implemented condition to exclude renaming when captioning. Uses flag defined in ArgSpec.
__key__ = "db7ee39184737f0d114f2ddc85500e38a7a414d7cc06f753116d223811afdf8b"
__version__ = "0.0.1"

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
    no_renaming = bool(args.disable_renaming)
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
                    elif no_renaming:
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
#‍ ∆eof
