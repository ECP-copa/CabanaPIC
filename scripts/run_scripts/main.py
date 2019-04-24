from git import Repo

repo_path = "../../"
r = Repo(repo_path)
repo_heads = r.heads # or it's alias: r.branches
repo_heads_names = [h.name for h in repo_heads]

kokkos_src = '/Users/bird/kokkos/'
kokkos_install = '/Users/bird/kokkos/build/install'
cabana_install = '/Users/bird/Cabana/build/build/install' # not a typo, it's in a dumb path

# Iterate over *local* git branches
for branch in repo_heads_names:

    # For each repo, check it out into a new folder and build it
    clone_path = './' + branch

    # look to see if the folder already exists:
    import os, shutil
    if os.path.isdir(clone_path):
        # if it does... delete it (!)
        shutil.rmtree(clone_path)

        # if it does... skip
        #continue

    cloned = Repo.clone_from(
        repo_path,
        clone_path,
        branch=branch
    )

    import subprocess
    subprocess.check_call(['./build_and_run.sh', clone_path, "g++", kokkos_install, cabana_install])

    print branch
