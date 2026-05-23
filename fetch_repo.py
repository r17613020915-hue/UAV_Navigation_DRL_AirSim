import requests

base_url = "https://api.github.com/repos/EthanRath/Backdoors-In-RL/contents"

def list_dir(path):
    url = f"{base_url}/{path}"
    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        items = r.json()
        for item in items:
            if item["type"] == "file":
                print(f"FILE: {item['path']}")
            else:
                print(f"DIR: {item['path']}")
                list_dir(item["path"])
    else:
        print(f"Failed to list {path}: {r.status_code}")

print("Repository structure:")
list_dir("")
