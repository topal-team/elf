## Configuration to build useful singularity images

To build an image, simply install [singularity](https://docs.sylabs.io/guides/3.5/user-guide/index.html#) and type:

```
sudo singularity build name.sif name.def
```

Images are available on PlaFRIM at /beegfs/aaguilam/images.

## Singularity images usage

The folder containing your code needs to be binded into the image. Otherwise, the image should be re-built every time the code is updated.

```
[srun] singularity exec --bind /path/to/code:/path/in/image --nv /path/to/image script.py
```

For example, to run with my updated nanotron code, I do:

```
singularity exec --bind /home/aaguilam/nanotron:/nanotron --nv /beegfs/aaguilam/images/nanotron.sif script.py
```

When using on PlaFRIM, don't forget to add ``#SBATCH --exclude=sirocco[01-06]`` to your script to avoid nodes with older gpus + cuda versions :)
