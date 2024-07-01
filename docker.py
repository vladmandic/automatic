import subprocess
import re
import yaml

platformDict = {
    "CUDA": {
        "11.8": "11.8.0",
        "12.0": "12.0.1",
        "12.1": "12.1.1",
        "12.2": "12.2.2",
        "12.3": "12.3.2",
        "12.4": "12.4.1",
        "12.5": "12.5.0",
    },
    "ROCm":{
        "5.5": "5.5.1",
        "5.6": "5.6.1",
        "5.7": "5.7.1",
        "6.0": "6.0.2",
        "6.1": "6.1.2",
    }
}

IMAGE = None

def get_image_ver(ver, platform):
    dict = platformDict[platform]
    ver = ".".join(ver.split(".")[0:2])

    dictKeys = sorted(dict.keys(), key=lambda x: float(re.sub(r"[^\d.]", "", x)))

    if ver < dictKeys[0]:
        raise ValueError(f"{platform} version is below the minimum required version. Detected Ver: {ver}, Min Ver Required: {dictKeys[0]}")
    elif ver > dictKeys[-1]:
        useVer = dictKeys[-1]
    else:
        useVer = max(version for version in dictKeys if version <= ver)

    return dict[useVer]


def check_cuda():
    try:
        smiRes = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    except:
        print("Nvidia SMI not detected")
        return

    cuRegex = r"CUDA Version: ([\d\.]+)"
    cuVer = re.search(cuRegex, smiRes)

    if cuVer:
        ver = get_image_ver(cuVer.group(1), "CUDA")
        return f"nvidia/cuda:{ver}-runtime-ubuntu22.04"
    else:
        print("Cant find CUDA version")


def check_rocm():
    try:
        hcRes = subprocess.check_output(['hipconfig --version'], shell=True).decode('utf-8')
    except Exception as e:
        print(e)
        print("ROCM not detected")
        return
    ver = get_image_ver(hcRes, "ROCm")
    return f"rocm/dev-ubuntu-22.04:{ver}"

IMAGE = check_cuda()
if not IMAGE:
    IMAGE = check_rocm()
if not IMAGE:
    IMAGE = "ubuntu:24.04"
print(IMAGE)

with open('docker-compose.yml', 'r') as file:
    data = yaml.safe_load(file)
data["services"]["webui"]["build"]["args"]["BASE_IMG"] = IMAGE
print(data)
with open('docker-compose.yml', 'w') as file:
    yaml.dump(data, file)
