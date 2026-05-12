import subprocess
import random
from datetime import datetime, timedelta
import os

# ==== CONFIG ====
start_date = datetime(2026, 4, 18)
end_date = datetime(2026, 5, 11)

AUTHOR_NAME = "bigben-dev777"
AUTHOR_EMAIL = "albertojohn20041103@gmail.com"

messages = [
    "fix bug",
    "refactor",
    "cleanup",
    "minor tweak",
    "improve logic",
    "adjust formatting",
    "optimize",
    "update comments",
    "small fix",
    "wip",
]


# ==== GET TRACKED FILES ====
def get_tracked_files():
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    files = result.stdout.strip().split("\n")
    return [
        f
        for f in files
        if os.path.isfile(f) and not f.endswith((".png", ".jpg", ".gif", ".exe"))
    ]


# ==== RANDOM TIME ====
def random_time(is_weekend):
    if is_weekend:
        hour = random.randint(10, 22)
    else:
        hour = int(random.gauss(15, 3))
        hour = max(9, min(hour, 23))
    return hour, random.randint(0, 59), random.randint(0, 59)


# ==== COMMITS PER DAY ====
def commits_per_day(is_weekend):
    if is_weekend:
        return random.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]
    else:
        return random.choices(
            [0, 1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.25, 0.2, 0.15, 0.1]
        )[0]


# ==== MODIFY FILE (subtle edits) ====
def modify_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if not lines:
            return False

        idx = random.randint(0, len(lines) - 1)
        choice = random.random()

        if choice < 0.4:
            lines[idx] = lines[idx].rstrip() + " \n"  # whitespace tweak
        elif choice < 0.8:
            lines[idx] = lines[idx].rstrip() + f"  # tweak {random.randint(1,100)}\n"
        else:
            lines.append(f"# touch {random.randint(1,1000)}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True
    except:
        return False


# ==== MAIN ====
files = get_tracked_files()

if not files:
    print("No tracked files found. Add files to git first.")
    exit(1)

current = start_date

while current <= end_date:
    is_weekend = current.weekday() >= 7

    # skip some days (simulate real gaps)
    if random.random() < 0.15:
        current += timedelta(days=1)
        continue

    num_commits = commits_per_day(is_weekend)

    for _ in range(num_commits):
        file = random.choice(files)

        if not modify_file(file):
            continue

        subprocess.run(["git", "add", file])

        hour, minute, second = random_time(is_weekend)
        commit_time = current.replace(hour=hour, minute=minute, second=second)

        date_str = commit_time.strftime("%Y-%m-%d %H:%M:%S")

        env = {
            "GIT_AUTHOR_DATE": date_str,
            "GIT_COMMITTER_DATE": date_str,
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
        }

        msg = random.choice(messages)

        subprocess.run(["git", "commit", "-m", msg], env=env)

    current += timedelta(days=1)

print("Done generating realistic commit history.")
