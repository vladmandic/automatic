import argparse
import subprocess
import os
import re

IMG = {
    "cuda": "nvidia/cuda:12.1.1-runtime-ubuntu22.04",
    "rocm": "rocm/dev-ubuntu-22.04:6.0.2",
    "cpu": "ubuntu:22.04",
}

# Arg parser utils
class MultilineHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__("", 2, 55, 200)
    def _split_lines(self, text, _):
        lines = text.splitlines()
        lines.append("")
        return lines
class VolumeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        dictionary = getattr(namespace, self.dest)
        data_dir = getattr(namespace, "data_dir")

        if ":" in values:
            key, value = values.split(':', 1)
            if data_dir and key == "Data": raise argparse.ArgumentTypeError('Volume "Data" is reserved, as --data-dir presents')
            if value:
                dictionary[key] = value
            elif key in dictionary:
                del dictionary[key]
        else:
            print(f"Invalid item '{values}'. Items must be in the format 'VOL_NAME:VOL_PATH'")
def nonEmptyString(value):
    if value == "":
        raise argparse.ArgumentTypeError("Cannot be empty")
    return value

# Initialize and expose args
argParser = argparse.ArgumentParser(conflict_handler='resolve', add_help=True, formatter_class=MultilineHelpFormatter)
argParser.add_argument('-n', '--name', type=str, default = os.environ.get("SD_CONTAINER_NAME","SD-Next"), help = '''\
Specify the name for the container
Default: SD-Next
''')
argParser.add_argument('-p', '--port', type=int, default = os.environ.get("SD_PORT",7860), help = '''\
Specify the port exposed by the container
Default: 7860
''')
argParser.add_argument('--compute', type=str, choices=['cuda', 'rocm', 'cpu'], default = os.environ.get("SD_CONTAINER_COMPUTE","cuda"), help = '''\
Specify the compute platform use by the container
Default: cuda
''')
argParser.add_argument('--data-dir', type=nonEmptyString, default = os.environ.get("SD_DATADIR",None), help = '''\
Specify the directory for SD Next data
Default: None
''')
argParser.add_argument('-v', '--volumes', action=VolumeAction, default={
    "Cache": "/root/.cache",
    "Python": "/python"
}, metavar='VOL_NAME:VOL_PATH', help='''\
Mount a volume to the container
This flag can be used multiple times for multiple volumes mount
If you want to remove default volume, type "-v DEFAULT_VOL_NAME:" (leave VOL_PATH as empty)
Default:
    Data:[Path Specified With --data-dir] (save SD Next data to volume)
        Reserved for --data-dir. Unless --data-dir is not provided, it will always set to --data-dir
    Python:/python (save the venv to volume - venv will be created at /python/venv)
    Cache:/root/.cache (save the cache generated by pip, uv, huggingface)
'''
)
argParser.add_argument('--no-volume', default = os.environ.get("SD_CONTAINER_NO_VOL",False), action='store_true', help = '''\
Disable volume mounting (including default volume)
Default: False
''')
args, _ = argParser.parse_known_args()
args = vars(args).values()

# Command runner
wd = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(wd, "../docker.log"), "a+")
log.truncate(0)
def cmdStream(cmd, msg=None, expectErr=False):
    print(msg or f"Running - {cmd}\n")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, cwd=wd)
    for line in iter(process.stdout.readline, ''):
        if not msg: print(line, end='')
        ansi_escape = re.compile(r'\x1b[^m]*m')
        line = ansi_escape.sub('', line)
        log.write(line)
        log.flush()
    log.write("")
    process.wait()
    if process.returncode != 0 and not expectErr:
        print("An error has occurred, check docker.log for the details")
        exit(1)
    print("")