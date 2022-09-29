import os
import shutil

def clear_ipynb_checkpoints():
    for root, dirs, files in os.walk('.'):
        for dir in dirs:
            if dir == '.ipynb_checkpoints':
                shutil.rmtree(os.path.join(root, dir), ignore_errors=True)

if __name__ == '__main__':
    clear_ipynb_checkpoints()