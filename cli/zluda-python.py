import os
import sys


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from modules.zluda_installer import find, load
    load(find())

    import torch
    print(f'Python with ZLUDA {sys.version}')
    print('Type "help", "copyright", "credits" or "license" for more information.')

    while True:
        print('>>> ', end='')
        try:
            exec(input(), {
                'torch': torch,
            })
        except Exception as e:
            print(f'{e.__class__.__name__}: {e}')
