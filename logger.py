from datetime import datetime

def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("pipeline.log", "a") as f:
        f.write(f"[{now}] {message}\n")
    print(message)