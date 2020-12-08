# Simulating and estimating neural connection probability

## General usage
### Installation
```
git clone https://github.com/seankmartin/SKMNeuralConnections
cd SKMNeuralConnections\Code
pip install -r requirements.txt
```

### Graphical user interface
To run the code to perform statistical estimations and graph simulations of neural connection probability,
the main entry point is `python Code\ui.py` to show a PYQt5 GUI like so

![GUI image](Code/assets/UIpic.png)


### Command line interface
One can use the command line interface, with configs stored in the configs folder.
This offers a bit more flexibility than the GUI, use the `-h` flag to show help.

For example:

```
cd Code
python cli.py -cfg tetrode_ca1_sub.cfg
```

### Code documentation
Documentation is available at [GitHub Pages](https://seankmartin.github.io/SKMNeuralConnections/)

### Citation
This software repository can be cited from Zenodo at https://doi.org/10.5281/zenodo.4311795. 

## Further usage
If you are looking to further contribute to this project, or verify the results, then please read on.

### Building code documentation
Minimal code documentation is available at Code\docs\neuroconnect.

They can be rebuilt with the following commands.

```
cd Code
pdoc neuroconnect --html -o docs -f
cd ..
```

### Blue brain resources
Please place in the Code\resources folder everything from [our OSF repository](https://osf.io/u396f/).

### Reproducing all the figures
To reproduce all the figures, it will probably take about half a day on a decent computer.
To do so, run the following command after downloading all the blue brain resources:

```
cd Code
python cli.py -f
```

### Profiling the code
The code can take a long time to run. To profile it for performance improvements, run

```
cd Code
python profile_this.py ARGS
```

Where `ARGS` are what you would pass to `cli.py`.