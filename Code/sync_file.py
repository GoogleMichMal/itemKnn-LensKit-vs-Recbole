import shutil
import os
import subprocess

def syncfile():
    folder_path = "Data"
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        else:
            print("Folder '{}' does not exist.".format(folder_path))
    except Exception as e:
        print("An error occurred:", e)


def refreshvs():
    task_name = "Code.exe"
    try:
        subprocess.run(["taskkill", "/F", "/IM", task_name], check=True)
    except subprocess.CalledProcessError as e:
        print(e)
    except Exception as e:
        print("An error occurred:", e)
