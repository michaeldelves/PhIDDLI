# PhIDDLI
Phenotype Image Data and Digital Learning Innovation (PhIDDLI) for evaluating and visualising fluorescence microscopy cellular phenotypes

## <B>Setting up PhIDDLI</B>

If you are running Ubuntu in a virtual machine (if you are already running Ubuntu natively then skip 1-5):
1. Download and install Virtual Box (https://www.virtualbox.org/) 

2. Download an iso file of Ubuntu (https://ubuntu.com/download/desktop) 

3. Install Ubuntu in Virtual Box (ensure it has access to at least 50 Gb of virtual disk drive and enough memory allocation to run).

4. Go through the Ubuntu installation process (several reboots of the virtual environment are needed) and install Virtual Box Guest Additions to set up a shared folder (Google instructions). This is not necessary, but having a shared folder between your virtual desktop and your real computer is helpful for moving data around.

5. Power down the virtual machine and restart. Now you have access to a shared folder in the file manager which you can pull data across.

6. Download PhIDDLI from [Github] (https://github.com/michaeldelves/PhIDDLI/) and open the folder. PhIDDLI runs by command line in the terminal window from this folder. Right mouse button anywhere in the folder to open a terminal window.

7. It is strongly advised to set up PhIDDLI in a virtual environment (venv) so that any changes needed to operate PhiDDLI do not interfere with any other installed programs. Type the following commands and press ENTER after each one:

> sudo apt-get install python3-pip

> sudo pip3 install virtualenv

> virtualenv -p /usr/bin/python3.8 venv

This is will create a new set of folders with your virtual environment named “venv” running Python version 3.8 (other versions appear to not install PhIDDLI correctly).

8. Activate the virtual environment by typing:
> . venv/bin/activate

9. Automatically install all the Python dependencies required for PhIDDLI by typing:

> pip install -r requirements.txt

After each one, lots of text will appear, dependencies will download and it will take a while to install each one. If all goes well, PhIDDLI is now ready to use. If not, the most common reason is that the incorrect version of Python was installed.

## <B>Running the example dataset</B>

1. If not already, activate the virtual environment:
> . venv/bin/activate

2. Navigate to cell-shape-analysis -> data -> src

3. Download the dataset (e.g. https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD633). The data is divided into different experiments, to start, download the "PhIDDLI Primary Screen" study component and unzip both the .tif (image) files and .xml (bounding box) files into the folder “src”.

4. Navigate back to cell-shape-analysis and open “List of commands.txt”. This contains all the steps that must be run in the PhIDDLI pipeline. Highlight all of it and copy it to the clipboard.

5. Open a terminal window in cell-shape-analysis. Right mouse button and paste the text in. PhIDDLI should start running immediately, but may need the user to press ENTER at the end to execute the final step. Each step must be executed in order, but once executed, data is saved and subsequent steps can be executed at a later time. For example, once the entire set of commands have been executed, all calculations are complete and data can be visualised directly at any later point by just running the last command:

> python3 scripts/visualize_embeddings.py data/plot.json

6. Once complete, the analysed data in csv format can be found in cell-shape-analysis/data/ as “export.csv”.

7. To interactively visualise the data, open the web browser and navigate to http://127.0.0.1:8050/. Note that it must be “http” not “https” otherwise it doesn’t load.



