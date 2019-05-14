
## Getting Started
 1. Create a new environment.
 2. Install dependencies
 3. Clone the repository 

```
  
  cd
  virtualenv -p python2.7 tfx-env
  source ~/tfx-env/bin/activate
  mkdir tfx; cd tfx

  pip install tensorflow==1.13.1
  pip install tfx==0.12.0
  git clone https://github.com/tensorflow/tfx.git
  cd ~/tfx/tfx/tfx/examples/workshop/setup
  ./setup_demo.sh

```
setup_demo will install the remaining dependencies and setup other requisites.

This will create a folder called *Airflow* in your home directory.

## Adding Data
1. Change the directory to ~/Airflow
2. Create a folder called east under the data folder 
3. Copy the tf record files into folder called east.

## Adding Dags

1. Delete the contents of the dag folder. 
2. Copy the contents from the tfx_east into the dag folder

## Running Code
1. Getting the dag to start requires starting the webserver and scheduler component to be triggered.


```
# Open a new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow webserver -p 8080

# Open another new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow scheduler

```
In a browser:

    Open a browser and go to http://127.0.0.1:8080

1. Graph will be listed in the grid view. 
2. Turn the toggle button to get the graph in ON state.
3. Press *trigger graph button*. 

This will start the processing of the graph.

## References

1.https://www.tensorflow.org/tfx/tutorials/tfx/workshop
