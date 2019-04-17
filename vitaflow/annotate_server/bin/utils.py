import glob
import os


def get_file_ext(x):
    if '.' in os.path.basename(x):
        return '.' + x.rsplit('.')[-1]
    else:
        return ''


def trim_file_ext(x):
    if '.' in x:
        return x.rsplit('.')[-2]
    else:
        return x


def check_n_create(full_path):
    if not os.path.isdir(full_path):
        print('Created missing dir `{}`!'.format(full_path))
        os.mkdir(full_path)
    else:
        # TODO: remove below
        # print('Using dir at `{}`!'.format(full_path))
        pass


def parser_folder(search_folder, exts=None):
    '''Return (base_filename, full_path_filename)'''
    bag = []
    for filename in glob.iglob(search_folder + '/*', recursive=True):
        file = os.path.basename(filename)
        if exts:
            for _ext in exts:
                if file.endswith(_ext):
                    bag.append((file, filename))
                    break
        else:
            bag.append((file, filename))
    return bag


def get_folder_config(dir_path, file_exts, trim_path_prefix):
    file_dict = {}
    _found_files = parser_folder(dir_path, file_exts)
    for file, full_path_filename in _found_files:
        url = full_path_filename.split(trim_path_prefix)[-1].lstrip(os.sep)
        file = os.path.basename(url)
        # print(url, file)
        file_dict[trim_file_ext(file)] = {
            'url': url,
            'file': file
        }
    return file_dict


def remove_extra_lines_from_string(inputstring):
    string_to_return = ""
    for line in inputstring.split("\n"):
        if len(line.strip()) > 0:  # Only add non empty lines to the string_to_return
            string_to_return = string_to_return + line
    return string_to_return


def runCmd(cmd):
    # string of a command passed in here
    from subprocess import run, PIPE
    string_to_return = str(run(cmd, shell=True, stdout=PIPE).stdout.decode('utf-8'))
    string_to_return = remove_extra_lines_from_string(string_to_return)
    return string_to_return


def run_parallel_cmds(list_of_commands):
    from multiprocessing.dummy import Pool
    # thread pool
    from subprocess import Popen, PIPE, STDOUT
    listofprocesses = [Popen(list_of_commands[i], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                       for i in range(len(list_of_commands))]

    # Python calls this list comprehension, it's a way of making a list
    def get_outputs(process):  # MultiProcess Thread Pooling require you to map to a function, thus defining a function.
        return process.communicate()[0]  # process is object of type subprocess.Popen

    outputs = Pool(len(list_of_commands)).map(get_outputs,
                                              listofprocesses)  # outputs is a list of bytes (which is a type of string)
    listofoutputstrings = []
    for i in range(len(list_of_commands)):
        outputasstring = remove_extra_lines_from_string(
            outputs[i].decode('utf-8'))  # .decode('utf-8') converts bytes to string
        listofoutputstrings.append(outputasstring)
    return listofoutputstrings
