from helper import argParser, cmdStream, IMG

args, _ = argParser.parse_known_args()

compute, noVol, Vol, port, name, data_dir = args.compute, args.no_volume, args.volumes, args.port, args.name, args.data_dir

CMD = ["./webui.sh", "--uv", "--listen"]
dockerArgs = ["-t", f"-p {port}:7860", f"--name {name}", '-e "venv_dir=/python/venv"']

if data_dir:
    dockerArgs.append(f'-e "SD_DATADIR={data_dir}"')
    Vol["Data"] = data_dir
Vol = Vol.items()

if compute == "cuda":
    dockerArgs.append("--gpus all")
    CMD.append("--use-cuda")
elif compute == "rocm":
    dockerArgs.extend(["--device /dev/dri", "--device /dev/kfd"])
    CMD.append("--use-rocm")
else: CMD.append("--use-cpu=all")

if not noVol:
    for vName, vPath in Vol:
        dockerArgs.append(f'-v "SD-Next_{vName}:{vPath}"')

dockerArgs= " ".join(dockerArgs)
CMD= " ".join(CMD)
print(f'''
Container Settings:
    Name: {name}
    Port: {port}
    Compute Platform: {compute}
    Volumes: {"Disabled" if noVol else ", ".join([f'{key} -> {value}' for key, value in Vol])}
    Docker Args: {dockerArgs}
    Container CMD: {CMD}
''')
cmdStream(f"docker container rm {name} -f", msg=f"Removing container named {name}...", expectErr=True)
cmdStream(f"docker image rm sd-next -f", msg="Removing image named sd-next...", expectErr=True)
cmdStream(f'docker build -t sd-next -f ./Dockerfile --build-arg="BASE_IMG={IMG[compute]}" ../', msg="Building Docker Image (might takes few minutes)...")
cmdStream(f'docker run {dockerArgs} sd-next {CMD}')