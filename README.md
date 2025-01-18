# Content

This repository contains the code for Machine Learning and Systems Lab and Selected Topics in Machine Learning Seminar at FAU Erlangen-Nürnberg.

# Machine Learning and Systems Lab

## [Lab 4](./ml_systems_lab/lab_04)
Building a Flask server for serving a pre-trained PyTorch model and showing the gradient activation map. The idea is to build this server in an embedded device, such as Raspberry Pi or Jetson Nano.

## [Lab 5](./ml_systems_lab/lab_05)
Fine-tuning a pre-trained PyTorch model on a custom dataset and comparing the accuracy with the original model using Flask server and POST requests.

## [Lab 6](./ml_systems_lab/lab_06)
Serving a .mar format model using TorchServe
## [Lab 7](./ml_systems_lab/lab_07)
Serving a .llamafile model as a server.
## [Lab 8](./ml_systems_lab/lab_08)
Generating images with a pre-trained StableDiffusion-1.5 model using HuggingFace pipeline.


# Selected Topics in Machine Learning

## [Diffusion Models beat GANs on Image Synthesis](./selected_topics_ml/diffusion_models_beat_gans)
Final presentation for the seminar. The presentation is about the paper "Diffusion Models Beat GANs on Image Synthesis" by Alex Nichol, Prafulla Dhariwal. The presentation includes the main ideas of the paper, the results, and the comparison with GANs.

Here's a small comparison of cosine and linear beta variance schedulers for the diffusion model forward process:
<div align="center">
    <div>
        <b>Cosine Scheduler</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <b>Linear Scheduler</b>
    </div>
    <img src="./selected_topics_ml/diffusion_models_beat_gans/diffusion_steps/cosine/cosine_noising.gif" width="250" alt="Cosine" />
    <img src="./selected_topics_ml/diffusion_models_beat_gans/diffusion_steps/linear/linear_noising.gif" width="250" alt="Linear" />
</div>

It seems like linear scheduler more aggressively noising the image compared to cosine scheduler. Either there is a bug in the code or the linear scheduler is not implemented correctly or this is the expected behavior.