import modal

app = modal.App("amazon-ml-test")


@app.function(cpu=8, memory=32768)
def check():
    import os, platform
    return f"Running on Modal: {os.cpu_count()} CPUs, Python {platform.python_version()}"


@app.local_entrypoint()
def main():
    print(check.remote())