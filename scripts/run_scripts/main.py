from git import Repo
import subprocess
import os, shutil

# I use this later to lazily generate an error with a message
class CustomError(Exception):
    pass

repo_path = "../../"
r = Repo(repo_path)
repo_heads = r.heads # or it's alias: r.branches
repo_heads_names = [h.name for h in repo_heads]

#kokkos_src = '/Users/bird/kokkos/'
#kokkos_install = '/Users/bird/kokkos/build/install'
#cabana_install = '/Users/bird/Cabana/build/build/install' # not a typo, it's in a dumb path

#platforms = ["Serial", "CPU", "GPU", "UVM"]
#platforms = ["Serial", "CPU", "GPU"]
platforms = ["CPU", "GPU"]
#platforms = ["GPU"]

CXX = "g++"
#arch = 'Volta70'
arch = 'Kepler35'

subprocess.check_call(['./timing_lib.sh'])

this_build_dir = 'build'

kokkos_dirs = {}
cabana_dirs = {}

home_dir = os.environ['HOME']

# Build Dependencies
# TODO: make this configurable
kokkos_root = os.path.join(home_dir,'kokkos')
cabana_root = os.path.join(home_dir,'Cabana')

# Check we can find Kokkos and Cabana
if not os.path.isdir(kokkos_root):
    raise CustomError("Can't find kokkos")
if not os.path.isdir(cabana_root):
    raise CustomError("Can't find Cabana")

# Copy Kokkos and Cabana to be inside this dir
def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def copy_if_safe(from_path, to_path):
    if not os.path.isdir(to_path):
        shutil.copytree(from_path, to_path)

# only copy if they don't exist already
kokkos_new = os.path.join(this_build_dir,'kokkos')
copy_if_safe(kokkos_root, kokkos_new)

cabana_new = os.path.join(this_build_dir,'cabana')
copy_if_safe(cabana_root, cabana_new)

# Build Dependencies
for plat in platforms:
    install_dir = "build-" + plat

    # Do Build
    print("build_kokkos.sh " + CXX + " " + kokkos_new + " " + install_dir + " " + plat + " " + arch)
    subprocess.check_call(['./build_kokkos.sh', CXX, kokkos_new, install_dir, plat, arch])

    print("./build_cabana.sh " + " " + CXX + " " + os.path.join(kokkos_new,install_dir,'install') + " " + cabana_new + " " + install_dir + " " + plat)
    subprocess.check_call(['./build_cabana.sh', CXX, os.path.join(kokkos_new,install_dir,'install'), cabana_new, install_dir, plat])

    # Save dirs, relative to root
    cabana_dirs[plat] = install_dir
    kokkos_dirs[plat] = install_dir


# Iterate over *local* git branches
for branch in repo_heads_names:
    for plat in platforms:

        print(plat)
        # TODO: throughout these scripts we assume ./instal is the install dir! abstract it.
        cabana_install = os.path.join( cabana_dirs[plat], 'install')
        kokkos_install = os.path.join( kokkos_dirs[plat], 'install')

        # For each repo, check it out into a new folder and build it
        #clone_path = './' + branch
        clone_path = os.path.join('./', this_build_dir, branch)

        # look to see if the folder already exists:
        if os.path.isdir(clone_path):
            # if it does... delete it (!)
            shutil.rmtree(clone_path)

            # OR if it does... skip
            #continue

        cloned = Repo.clone_from(
            repo_path,
            clone_path,
            branch=branch
        )

        pwd = os.getcwd()

        kokkos_full_path = os.path.join(pwd, kokkos_new, kokkos_install)
        cabana_full_path = os.path.join(pwd, cabana_new, cabana_install)
        print("kk full path " + kokkos_full_path)

        print("./build_and_run.sh " +  clone_path + " g++ " + kokkos_full_path + " " + cabana_full_path + " " + plat)
        subprocess.check_call(['./build_and_run.sh', clone_path, "g++", kokkos_full_path, cabana_full_path, plat])

        print branch
