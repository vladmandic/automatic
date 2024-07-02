from helper import argParser, cmdStream

args, _ = argParser.parse_known_args()

compute, noV, V, port, name = args.compute, args.no_volume, args.volumes, args.port, args.name

dockerArgs = ["-t", f"-p {port}:7860", f"--name {name}", '-e "venv_dir=/python/venv"']

if compute == "cuda": dockerArgs.append("--gpus all")
if compute == "rocm": dockerArgs.append("--device /dev/dri --device=/dev/kfd")

if not noV:
    for vName, vPath in V.items():
        dockerArgs.append(f'-v {("SD-Next_" if not vName =="SD-Next" else "")+vName}:{vPath}')

dockerArgs= " ".join(dockerArgs)
print(f'''
Container Settings:
    Name: {name}
    Port: {port}
    Compute Platform: {compute}
    Volumes: {"Disabled" if noV else ", ".join([f'{key} -> {value}' for key, value in V.items()])}
    Args: {dockerArgs}
''')
cmdStream(f"docker container rm {name} -f", msg=f"Removing container named {name}...", expectErr=True)
cmdStream(f"docker image rm sd-next -f", msg="Removing image named sd-next...", expectErr=True)
cmdStream(f"docker build -t sd-next -f ./{compute}.Dockerfile ../", msg="Building Docker Image (might takes few minutes)...")
cmdStream(f'docker run {dockerArgs} sd-next')