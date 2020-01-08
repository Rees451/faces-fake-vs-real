import os
from shutil import copyfile
import sys


def sort_files(source, dest, as_links):

    real_dir = f'{source}/real_and_fake_face/real'
    fake_dir = f'{source}/real_and_fake_face/fake'

    # Left eye
    criteria_copy(real_dir, fake_dir, dest, 'left_eye', as_links)
    # Right eye
    criteria_copy(real_dir, fake_dir, dest, 'right_eye', as_links)
    # Mouth
    criteria_copy(real_dir, fake_dir, dest, 'mouth', as_links)
    # Nose
    criteria_copy(real_dir, fake_dir, dest, 'nose', as_links)


def criteria_copy(real_dir, fake_dir, dest, criteria, as_links):

    if as_links is True:
        lin = '_links'
    else:
        lin = ''

    real_dest = f'{dest}/real_and_fake_face_ovr{lin}/' + criteria + '/real'
    fake_dest = f'{dest}/real_and_fake_face_ovr{lin}/' + criteria + '/fake'

    # Copy photoshoped files
    copy_files(fake_dir, fake_dest, real_dest, criteria, as_links)

    # Copy real files
    copy_files(real_dir, real_destination=real_dest, as_links=as_links)


def copy_files(source,
               fake_destination=False,
               real_destination='',
               criteria=None,
               as_links=True):

    list_dir = os.listdir(source)
    list_dir = [i for i in list_dir if '.jpg' in i]

    if not os.path.exists(real_destination):
        os.makedirs(real_destination)

    if not os.path.exists(fake_destination):
        os.makedirs(fake_destination)

    for file in list_dir:

        if criteria == 'left_eye':
            i = 0
        elif criteria == 'right_eye':
            i = 1
        elif criteria == 'nose':
            i = 2
        elif criteria == 'mouth':
            i = 3
        else:
            i = 8

        if file[-8 + i] == '1' and fake_destination is not False:
            destination = os.path.abspath(fake_destination + '/' + file)
        else:
            destination = os.path.abspath(real_destination + '/' + file)

        source_file = os.path.abspath(source + '/' + file)

        if as_links is False and not os.path.exists(destination):
            copyfile(source_file, destination)
        elif as_links is True and not os.path.exists(destination):
            os.symlink(source_file, destination)


def check_numbers(source, dest, criteria, as_links):

    if as_links is True:
        lin = '_links'
    else:
        lin = ''

    source1 = f'{source}/real_and_fake_face/real'
    source2 = f'{source}/real_and_fake_face/fake'
    destination1 = f'{dest}/real_and_fake_face_ovr{lin}/' + criteria + '/real'
    destination2 = f'{dest}/real_and_fake_face_ovr{lin}/' + criteria + '/fake'

    original = os.listdir(source1) + os.listdir(source2)
    original = len([i for i in original if '.jpg' in i])
    copied = os.listdir(destination1) + os.listdir(destination2)
    copied = len([i for i in copied if '.jpg' in i])

    assert original == copied
    return copied


if __name__ == '__main__':

    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = './ml-database'

    if len(sys.argv) > 1:
        dest = sys.argv[2]
    else:
        dest = './data/processed'

    # Sort files
    as_links = True
    sort_files(source, dest, as_links)

    # Check numbers of copied files
    number1 = check_numbers(source, dest, 'left_eye', as_links)
    number2 = check_numbers(source, dest, 'right_eye', as_links)
    number3 = check_numbers(source, dest, 'mouth', as_links)
    number4 = check_numbers(source, dest, 'nose', as_links)

    assert number1 == number2 == number3 == number4
    print(f'Copied {number1} images successfully')
