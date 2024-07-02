from helper import argParser, cmdStream

args, _ = argParser.parse_known_args()

dockerArgs = ["-it", f"-p {args.port}:7860", f"--name {args.name}"]

if args.compute != "cpu": dockerArgs.append("--gpus=all --device /dev/dri")

if not args.no_volume:
    for name, path in args.volumes.items():
        dockerArgs.append(f'-v {("SD-Next_" if not name =="SD-Next" else "")+name}:{path}')

cmdStream(f"docker build -t sd-next -f ./{args.compute}.Dockerfile ../")

cmdStream(f'docker run {" ".join(dockerArgs)} sd-next')