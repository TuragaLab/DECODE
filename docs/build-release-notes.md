# DECODE and SplinePSF build and release instructions

These instructions are for building and releasing the DECODE and SplinePSF repositories.

## Current status

Documented and working:
- pip build for Colab support
- conda build and Anaconda Cloud upload
- automated builds and uploads with GitHub Actions

## History
- September 2020: first draft
- October 2020: added conda build and Anaconda cloud instructions
- March 2021: added GitHub Actions automated build information


# Release overview

Note that the release processes for the two repositories are independent.  There may be version dependencies in the actual code, but the machinary that generates the releases is independent.

Here's the general procedure. Items listed in **bold** are section titles that appear below:

- **Finalize repositories**
- Automated build:
    + **Build automatically with GitHub Actions**
        * **Create GitHub release**
        * **Google Colab changes**
        * That should be it, which is the point of the automation!  The only manual step is updating Google Colab, if you want to use the latest version in that environment.
        * Optional: monitor the build progress and outcome from the GitHub Actions tab of each repo
- (or) Release manually:
    + **Build with conda**
        * **Build and upload SplinePSF**
        * **Build and upload DECODE**
    + **Build wheels for pip install**
        * **GitHub release**
        * **Google Colab changes**
- Verify and test
    + If using automated builds, verify they completed without error on the GitHub Actions page for each repo
    + Navigate to the Anaconda Cloud account and check that all expected files were uploaded
        * This is especially important for the automated builds; I have seen transient upload errors with Anaconda Cloud where some uploads succeeded and others failed
    + Optionally, do a fresh install from Anaconda Cloud and run tests

## Finalize repositories

For release via any method:
- push all code to repos
- be sure metadata is up to date
- perform all desired tests
- update versioning
    + version number appears in setup.py, meta.yaml, and (DECODE only) `__init__.py` files
    + however, you should use the "bump2version" tool 
        * see https://github.com/c4urself/bump2version/#installation
        * you will need to `pip install --upgrade bump2version` and run the tool with the new version info
    + push version changes
- tag the desired release commits in both repos with the version name/number
    + which should match what you assigned in the "bump2version" step!
    + and push the tags to GitHub if you do it locally


# Manual build and release

## Build with conda

### First time only
These notes are for all platforms. See platform-specific notes below.

- install conda or miniconda if it's not already installed
- create a conda env just for builds (here using name "build_clean"); include the build and install tools
    + `conda create --name build_clean conda-build anaconda-client`
    + this env is mostly empty for now (Python plus the build tools); the builds will install dependencies listed in the meta.yml files

### Conda build and Anaconda cloud upload

**General notes applying to both repos**
- there's a `conda` folder in each repo; cd into that folder to do each build
- the "conda-forge" channel is the standard source of all the normal libraries the projects depend on; some builds require other channels as well
- the output will be found in the `build_clean/conda-bld` directory in the conda environments directories
- the build will create multiple versions, if configured that way (eg, for multiple Python versions)
- the build can take 10-20 minutes (for 3 Python versions currently); compiling is fairly quick, but each build creates and installs an isolated conda environment corresponding to the various Python versions
- the text output of the build will give you the location of the output files and the command to upload to Anaconda cloud
    + note that you will need to add the organization name to the commands
- `anaconda login` before you do any Anaconda cloud operations
    + your login token will be cached for a while (a week?)
    + to upload a _new_ package, you must have owner privileges in the organization
    + to update an _existing_ package, you need only have read-write privileges
- after everything is done, you may optionally clean up temporary old files in each build; in the same directory as the conda build was done:
    + `conda build purge`
- `anaconda upload` can fail silently; there's a bug in the client where it swallows errors it should display; you can hand-patch the client to show those errors (ask Don how if needed)

**Build and upload SplinePSF**
Build and upload spline before decode, as decode depends on spline.
- `conda activate build_clean`
- Windows only: run compiler variable setup script (see below)
- `cd SplinePSF/dist_tools/conda`
- `git checkout (branch to release)`
- `conda-build -c conda-forge spline`
- `anaconda login` 
- `anaconda upload -u Turagalab /path/to/output/file/spline-blah-blah.tar.bz2`
    + repeat for each version that was built

**Build and upload DECODE**
- `conda activate build_clean`
- `cd DECODE/conda`
- `git checkout (branch to release)`
- `conda-build -c turagalab -c pytorch -c conda-forge decode`
- `anaconda upload -u Turagalab /path/to/output/file/decode-blah-blah.tar.bz2`
    + repeat for each version that was built

At this point, you can check https://anaconda.org/Turagalab/ and verify that files were uploaded as desired. You may also want to remove old versions, if applicable.

### Platform-specific notes

**Linux**
- so far I (djo) have only succeeded in using turagas-ws1 for the Linux build
- I was not able to build on a Scientific Linux 6.5 computer
- I was not able to build on a cluster node (SL 7.3, late 2020)
- presumably needs cudatoolkit (Lucas says v10.1)


**Windows**
- conda:
    + for Windows, I found that putting conda on the %PATH% was problematic 
        * got some SSL errors when trying to install packages
    + I started a terminal session from within Anaconda Navigator, and it worked better, as the Internet said it would

- compilers:
    + install MS Visual Studio 2019; free community edition is fine
    + install cuda toolkit (Lucas says v10.1)

- path setting script from MS VS
    + after activating the environment but before doing the build, execute the Windows compiler variable setup script; for my installation, that was: 
        * "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64


**Mac**
- need Mac dev tools installed
- no cuda for Mac (do any Macs have nVidia cards anymore?)
- Mac build can be tricky and require a specific version of the Mac OS SDK; the version that shipped with XCode 11.5 is known to work (used in the GitHub Actions version of the build)

## Build wheels for pip install
This build is a hybrid. We will be using conda to prepare the environment, but we will be using pip to create the wheel file. 

- get the code: 
    + either git clone the two repositories somewhere convenient
    + or git pull in existing repos so you have the latest changes and tags
- prepare environment if it hasn't been done
    + this is needed for building SplinePSF
    + install miniconda if it isn't installed 
    + `cd SplinePSF/`
    + `conda env create -f python/environment.yml`
        * the environment file specifies the name of the env (spline_dev as of this writing); it should be used in the next step
- `conda activate spline_dev`
- build SplinePSF wheel    
    + `cd SplinePSF/python`
    + `python setup.py bdist_wheel`
        * the wheel will be in `python/dist`
        * if you want the bare library instead of the wheel, do `python setup.py build`
- build DECODE wheel
    + `cd DECODE`
    + `python setup.py bdist_wheel`
        * the wheel will be in `dist/`
        * it is pure Python even though the command to build the wheel says "bdist"

    
## GitHub release
- in each repo on GitHub, choose "Create a new release" from right-hand column
- choose the release tag 
- name the release, preferably the same as the tag
- in release dialog, upload the two wheel files
- click "create" (or "update", I forget)
- note: you can edit the release after it's created, including replacing the wheel files


## Google Colab changes
- add install lines (using URLs from above release):
    + !pip install (SplinePSF URL)
    + !pip install (DECODE URL)
- installation order is _not_ actually critical; DECODE depends on SplinePSF, but that dependency is not currently enforced in the build
    + however, if you try to "import decode" before SplinePSF is installed, it will fail in mysterious and frustrating ways (ask me how I know that...)
- test


## testing Anaconda Cloud uploads
To test the uploads in a fresh environment:
- `conda create -n test_temp -c turagalab -c pytorch -c conda-forge decode`
    + yes, this takes a long time; resist the urge to reuse a previous testing environment if you've made any changes to required packages, etc.
- `conda activate test_temp`
- `git clone https://github.com/TuragaLab/DECODE.git`
- optional: `git checkout (version you want to test if not master)`
- `cd DECODE`
- `pytest decode/test`

Running the tests does create some temporary files. Delete them if you don't want to clutter your repo (we should add them to .gitignore)


# Build automatically with GitHub Actions

GitHub Actions is GitHub's system for doing automatic testing, continuous integration/deployment, and related things.

## What it does

For both the DECODE and SplinePSF repositories, when a release is created in GitHub:

- the conda build is run
    + for both repos, Python 3.6 - 3.9 are built
        * this is part of the conda build; it does not use GitHub Actions matrixing
    + for SpinePSF, three platforms are built (Linux, Mac, Windows)
    + for DECODE, the pure Python noarch build is done
- all builds are uploaded to the Turagalab Anaconda Cloud account
- the pip build is run; for both repos, the resulting files are uploaded to the GitHub release, in support of Colab
- that's a lot of builds!  
    + the platform builds will be done in parallel
    + the Python version builds are done in sequence for each platform
    + the GitHub builds are slower than builds done on local machines, and there may be queueing times; use the GitHub Actions dashboard to monitor progress

## Push vs release & testing

For DECODE, testing is done in separate workflows.  The "build-upload" workflow only triggers on release.

SplinePSF tests are currently contained in DECODE and are run when DECODE is tested. When SplinePSF is pushed, released, or a pull request is received, the conda build is triggered on all three platforms as a minimal "does it build" test. The uploads and the pip build are not done on push or pull request.

## How it works
- in each repo, the .github/workflows folder contains a `build-upload.yml` file containing the workflow
    + the workflow file describes what is run as well as what triggers the build
- on the GitHub page for each repository, the Actions tab shows all workflows and lets you manage them, enable or disable them, and see log output when they are run
- it runs the same builds that can be run by hand on local machines
- if a build fails, (some person) will get a GitHub notification, using whatever method they have set in their profile


## Create GitHub release

The automated builds are triggered by the creation of a GitHub release.
- tag the desired release
- on the GitHub page for the desired repo, click "Releases" at right, then "Draft a new release"
- enter the tag and metadata and "Publish release"; this will trigger the "build-upload.yml" workflows
    + later edits to the release will re-trigger the workflow
    + if you create the release and save a draft, the workflow will not trigger; you need to publish the release for the workflow to trigger


## Anaconda Cloud management

Anaconda Cloud requires a separate login to allow uploads. GitHub Actions has a "secrets" system to manage things like these credentials. They are encrypted on the client before upload, and when accessed properly, secrets will not appear in any logging that GitHub does. 

Fortunately, Anaconda provides a token system for this purpose, so no personal account credentials need to be exposed. A token will need to be generated via Anaconda and uploaded to GitHub. This step will need to be repeated regularly, as the token has a finite lifespan (one year default). If at some point the token becomes less-than-secret, it can be revoked.

It's easiest to manage Anaconda Cloud tokens from the Anaconda web UI. Be sure you have switched to the "Turagalab" account in the upper-right corner, then choose "Settings", then "Access" from the list at left.

You can also manage tokens using the `anaconda` command-line client (installed via `conda install anaconda-client`, which is done in the build environment as described above).

**NOTE:** Tokens have an expiration date!  This should be obvious when it happens; the workflow will fail, and you will need to recreate and reupload the token, then retrigger the workflows.

### Create, revoke, or manage tokens

As noted above, it's easiest to create the token from the website. Switch to the Turagalab account, choose "Settings" then "Access". You will likely need to reauthenticate once or twice during the process of managing tokens.

Don created the following token on December 16, 2020:
- name: GitHubActionsToken
- strength: strong
- scopes: read and write access to API site
    + this is what the `anaconda` client needs to do its upload
- expiration date: 2022/01/31
    + default is a year, but I lengthened it a bit so it didn't fall near holidays

You can view or revoke the token from the website as well. 

**CAREFUL:** Be careful with this token. If you possess this token, you can delete all files in the account. Note, though, that its scope is restricted to file management. The token will not allow you to manipulate user permissions. Therefore this token should not be able to lock you out of the account (and, eg, prevent you from revoking it).


### Upload token to GitHub "secret"

Note that this must be done for each repo individually. You must have admin access to the repository to add a secret. From the repo's "Settings" page, choose "Secrets", and you can perform the operations there (add/remove/etc.).

For both repos, I used the name "TURAGALAB_ANACONDA_TOKEN" (December 10, 2020).



