# End-to-End Deep Learning for Self-Driving Cars Car using Pytorch Lightning

## Simullator installation

Install the simulator following:
https://kaigo.medium.com/how-to-install-udacitys-self-driving-car-simulator-on-ubuntu-20-04-14331806d6dd

Run on GPU by:

`RUN_GRAPH=true ~/Downloads/beta_simulator_linux/beta_simulator.x86_64`

## Collect dataset
1. Start the simulator
2. Select TRAINING MODE
3. Click on the RECORD button, the first time you will be asked where to place the recorded data, place it under `./data` in the root folder of this repo
5. Click on RECORD again, this second time the stet of the button will change to RECORDING, when puased it will save all the data in buffered

## Train on collected dataset
Start the training by:

```bash
python3 train.py
```

## Use the trained model to controll the simulated car
1. Start the simulator
2. Select AUTONOMOUS MODE
3. Run the script that process the video feed and drives the car

```bash
python3 drive.py
```
