from helper import argParser, cmdStream

args, _ = argParser.parse_known_args()

dockerArgs = ["-it", f"-p {args.port}:7860", f"--name {args.name}"]

if args.compute != "cpu": dockerArgs.append("--gpus=all --device /dev/dri")

if not args.no_volume:
    for name, path in args.volumes.items():
        dockerArgs.append(f'-v {("SD-Next_" if not name =="SD-Next" else "")+name}:{path}')

cmdStream(f"docker build -t sd-next -f ./{args.compute}.Dockerfile ../")

cmdStream(f'docker run {" ".join(dockerArgs)} sd-next')

# # Example usage
# stream_command_output('docker run -it --device /dev/dri -v SD-Next:/workspace -v SD-Next_Venv:/python -v SD-Next_Cache:/root/.cache -p 7860:7860 --gpus=all --name "SD-Next" sd-next')