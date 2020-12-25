# AirSim - Game of Drones, NeurIPS 2019
This repository is my own exploration of data engineering and AI-based perception for autonomous systems. Using the binaries from the competition organizers, this serves as good practice for using AirSim and finding potential strategies for future competitions and applications. As graphics get more and more photorealistic, simulators may be the way to go in order to get the large amount of data needed to train neural networks.  

Due to life obligations, my team was unable to compete. I was also the only engineer for perception on my team. Although the competition ended long ago, I'm continuing for my own development.

## Dataset Generation
Using traditional computer vision methods to identify flyable regions of gates can be computationally costly and too dependent on selecting the correct parameters. Using a neural network based method with masks can be more robust.

Since AirSim uses the Unreal engine, it would be ideal to edit the gate assets to have invisible flyable regions so the labels can be generated. However, the assets in the competition binaries are not accessible. Given the gate positions relative to the drone and their geometry, flyable regions can be projected to the camera view, allowing additional labeling for images.

The dataset generation can support labelling for instance segmentation.

## Neural Network Training
The current focus is to use semantic segmentation as a proof of concept. Depth perception and instance segmentation will be the next steps.

Since realtime processing is needed for the high speeds that drones travel, currently Fast-SCNN is being used. Since the authors did not publish code for Fast-SCNN, an independent implementation to replicate their efforts should be done to minimize potential errors. That effort is being done [here](https://github.com/rachthree/fast_scnn).

## References
* [Game of Drones, NeurIPS 2019](https://www.microsoft.com/en-us/research/academic-program/game-of-drones-competition-at-neurips-2019/)
* [AirSim NeurIPS Binaries](https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/)