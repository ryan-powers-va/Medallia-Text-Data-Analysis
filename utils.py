import os
from hashlib import sha256
import stat
import shutil

# Caching function to establish cache directory. 
def get_cache_file_path(comment, task, model, prompt):
    key = f"{comment}_{task}_{model}_{prompt}"
    hashed_key = sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_directory, f"{hashed_key}.json")

# Cache clearing function. 
cache_directory = r"cache"
def force_delete_readonly(func, path, exc_info):
    # Change the permission and reattempt removal
    os.chmod(path, stat.S_IWRITE)
    func(path)

# def clear_cache():
#     try:
#         # Delete directory, handling read-only files
#         shutil.rmtree(cache_directory, onerror=force_delete_readonly)
#         print("Cache cleared successfully.")
#     except FileNotFoundError:
#         print("Cache directory does not exist. Nothing to clear.")
#     except PermissionError:
#         print("Permission denied. Ensure the directory is not open in another program.")
#     except Exception as e:
#         print(f"An error occurred while clearing the cache: {e}")

# clear_cache()


# Predefined tags
TAGS = [
    "Integration", "Ease of use", "Early pop up", "Findability/Navigation", "Sign in/access"
]

# TAGS = [
#     "Triage Group", "Error", "Integration", "Ease of use", "Other", "Early pop up", "Findability/Nav", "Sign in/access", "Answered Question"
# ]


# TAGS = [
#     "Triage Group", "Unrelated to VA.gov", "Bug", "Integration", "Ease of use",
#     "Benefits", "Feature Request", "Supplies", "Other", "Mixed Status", "Can't Reply",
#     "Early pop up", "Missing Rx", "Findability/Navigation", "Sign in or access",
#     "Content", "Sort", "Answered Question", "Can't Refill", "Page Length"
# ]
