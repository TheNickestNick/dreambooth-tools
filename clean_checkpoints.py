import os

def clear_ipynb_checkpoints():
    for root, dirs, files in os.walk('.'):
        for name in dirs:
            if name == '.ipynb_checkpoints':
                os.remove(os.path.join(root, name))

if __name__ == '__main__':
    clear_ipynb_checkpoints()