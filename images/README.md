## Configuration to build useful singularity images

To build an image, simply install [singularity](https://docs.sylabs.io/guides/3.5/user-guide/index.html#) and type:

```
sudo singularity build name.sif name.def
```

Images are available on PlaFRIM at /beegfs/aaguilam/images.

When using on PlaFRIM, don't forget to add ``#SBATCH --exclude=sirocco[01-06,10-15,17,24-25]`` to your script to avoid nodes where singularity is missing :)
