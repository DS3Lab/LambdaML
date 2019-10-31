import os
import numpy as np

def random_files(num_files, path):
    for i in np.arange(num_files):
        w = np.random.rand(2, 3)
        np.savetxt(path + "/" + str(i), w)


def create_file(path):
    """return a new file object ready to write to"""
    print("creating file {}".format(path))
    return open(path, 'w')


def split_file_with_info(src_path, dst_path, num_files):
    line_number = 0

    src_file = open(src_path, "r")
    print("Splitting {} into {} files".format(src_path, num_files))

    dst_file = []
    for i in np.arange(num_files):
        file_name = "{}_{}".format(i, num_files)
        dst_file_name = os.path.join(dst_path, file_name)
        dst_file.append(create_file(dst_file_name))

    for line in src_file:
        file_index = line_number % num_files
        dst_file[file_index].write(line)
        line_number += 1

    for file in dst_file:
        file.close()


def create_trigger_file(path, num_files):
    for i in np.arange(num_files):
        file_name = "{}_{}".format(i, num_files)
        dst_file_name = os.path.join(path, file_name)
        f = open(dst_file_name, 'w')
        f.write("test")
        f.close()


def merge_files(src_files, dst_file):
    dst_file = open(dst_file, 'w')
    for src_name in src_files:
        src_file = open(src_name, 'r')
        for line in src_file:
            dst_file.write(line)
        src_file.close()
    dst_file.close()


if __name__ == "__main__":
    src_file = "C:\\Users\Jiawei\\Downloads\\dataset\\rcv1\\rcv1_libsvm"
    dst_dir = "C:\\Users\Jiawei\\Downloads\\dataset\\rcv1\\splits"
    split_file_with_info(src_file, dst_dir, 5)
    #create_trigger_file(dst_dir, 50)
    #src_files = ["C:\\Users\\Jiawei\\Downloads\\dataset\\rcv1\\rcv1_train.binary",
    #             "C:\\Users\\Jiawei\\Downloads\\dataset\\rcv1\\rcv1_test.binary"]
    #dst_file = "C:\\Users\Jiawei\\Downloads\\dataset\\rcv1\\rcv1_libsvm"
    #merge_files(src_files, dst_file)
