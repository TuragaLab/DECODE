import os
import sys
import glob
import subprocess
import time
import pathlib
from pathlib import Path


class Candidate:
    """
    Manages a candidate (folder) which should contain a .json.
    This simple script then replaces the folder name etc.
    """
    def __init__(self, abspath, config_file):
        """
        
        :param abspath: absolute path to the folder 
        :param config_file: 
        """
        self.abspath = abspath
        self.config = config_file
        
        if not Path(self.abspath).is_dir():
            raise ValueError("Candidate must be folder not file.")

    @property
    def candidate(self):
        return pathlib.PurePath(self.abspath).name

    @property
    def config_abspath(self):
        return os.path.join(self.abspath, self.config)

    @property
    def config_noext(self):
        return os.path.splitext(self.config)[0]

    @property
    def parent(self):
        return Path(self.abspath).parent

    @property
    def extension(self):
        return Path(self.config).suffix

    def get_logs(self, abs=False):
        """
        Consturct the filename for log and error log.
        :param abs:
        :return:
        """
        log_suffix = '_log.log'
        err_suffix = '_err.log'
        if abs:
            return os.path.join(self.abspath, self.config_noext + log_suffix), \
                   os.path.join(self.abspath, self.config_noext + err_suffix)
        else:
            return os.path.join(self.config_noext + log_suffix), os.path.join(self.config_noext + err_suffix)

    def flag_candidate(self, flag):
        """
        Add a flag to the candidates folder name
        :param flag:
        :return:
        """
        name_new = flag + self.candidate.__str__()
        abspath_new = os.path.join(self.parent, name_new)
        os.rename(self.abspath, abspath_new)
        self.abspath = abspath_new


def watch_folder(path, keyword='train'):
    """
    Watches a given folder
    :param path: path of the parent folder to watch
    :param keyword: prefix of folders to watch for
    :return: None when there are no unseen folders and a path if there are
    """

    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def get_config(path):
        extension = 'json'
        config_files = [f for f in os.listdir(path) if f.endswith('.' + extension)]

        if config_files.__len__() == 0:
            return
        else:
            return config_files[0]

    dirs = get_immediate_subdirectories(path)
    for i in range(dirs.__len__()):
        if dirs[i][:keyword.__len__()] == keyword:
            config_file = get_config(path + '/' + dirs[i])
            return dirs[i], config_file

    return None, None


def launch_process(cmd_run, out_file=None, err_file=None):
    if out_file is not None:
        with open(out_file, "w+") as out, open(err_file, "w+") as err:
            p = subprocess.Popen(cmd_run, shell=True, stdout=out, stderr=err)
    else:
        p = subprocess.Popen(cmd_run)

    return p


if __name__ == '__main__':
    folder_to_watch = '/Users/lucasmueller/Desktop/watch_folder'
    working_directory = '/Users/lucasmueller/Repositories/DeepSMLM'
    command_part = 'source activate deepsmlm_deployed;'
    command_part += 'python -m deepsmlm.neuralfitter.train_wrap -p '

    while True:
        os.chdir(working_directory)
        subfolder, candidate = watch_folder(folder_to_watch)
        if candidate is None:
            time.sleep(15)
        else:
            can = Candidate(os.path.join(folder_to_watch, subfolder), candidate)
            can.flag_candidate('running_')
            command_run = command_part + can.config_abspath
            log_file, err_file = can.get_logs(abs=True)
            p = launch_process(command_run, log_file, err_file)
            p.wait()
            can.flag_candidate('finished_')
