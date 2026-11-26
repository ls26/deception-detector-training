def sanitize_cache_path(path: str) -> str:
    """Sanitize a filesystem path by replacing ':' in each path component.

    Dataset names like ``local:diverse_deception`` are valid identifiers but
    cannot be used as directory names on all filesystems.
    """
    if not path:
        return path
    parts = path.split("/")
    sanitized_parts = [part.replace(":", "-") for part in parts]
    if path.startswith("/"):
        return "/" + "/".join(sanitized_parts[1:] if parts[0] == "" else sanitized_parts)
    return "/".join(sanitized_parts)
