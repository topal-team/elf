## Setup on Jean-Zay

### With a virtual environment

It is better to create the environment in your WORK directory, as the home one is very limited.

```bash
module load python/3.11.5 # (you need python >= 3.10)
python -m venv venv
source venv/bin/activate

# I recommend installing the nightly version
# The V100 partition has cuda 12.8, even if no module exists with this version
pip install --no-cache --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --no-cache -r requirements.txt
```

### With a container

An image is available in ``elf-dev/images/pipe.def``.\
You can to build it with ``sudo apptainer build pipe.sif pipe.def`` on your local machine.\
Then, you can scp it to Jean-Zay and add it to the directory of allowed images with ``idrcontmgr cp pipe.sif``.\
To use it, you can do ``singularity exec [--nv] $SINGULARITY_ALLOWED_DIR/pipe.sif [your command]``. ``--nv`` is needed to use GPUs. Don't forget to ``module load singularity/3.8/5`` beforehand.

```
! This will not work for partitions with ARM CPUs, such as the H100 partition !
```

### METIS and DagP

```bash
module load gcc/14.2.0 ;
cd $WORK ;
mkdir packages ;
cd packages ;
git clone https://github.com/KarypisLab/GKlib.git ;
cd GKlib ;
make config prefix=$WORK/packages ;
make ;
make install ;
cd .. ;
git clone https://github.com/KarypisLab/METIS.git ;
cd METIS ;
make config prefix=$WORK/packages ;
make ;
make install ;
```

and then ``export PATH=$WORK/packages/bin:$PATH`` in your ``.bashrc``.