<div align="center">

# Sound Field Reconstruction via Repeated Refinement: SR3-Based Room Transfer Function Interpolation Using Image Super-Resolution

[Riccardo Passoni](https://openresearch.surrey.ac.uk/esploro/profile/riccardo_passoni/overview)<sup>1</sup>, [Stefano Damiano](https://www.kuleuven.be/wieiswie/en/person/00148509)<sup>2</sup>, [Philip J.B. Jackson](https://www.surrey.ac.uk/people/philip-jackson)<sup>3</sup>, [Toon von Waterschoot](https://www.kuleuven.be/wieiswie/en/person/00042022)<sup>2</sup>, [Enzo De Sena](https://www.surrey.ac.uk/people/enzo-de-sena)<sup>1</sup>

<sup>1</sup> Institute of Sound Recording (IoSR), University of Surrey, Guildford, UK <br>
<sup>2</sup> Dept. of Electrical Engineering (ESAT-STADIUS), KU Leuven, Leuven, Belgium <br>
<sup>3</sup> Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey, Guildford, UK <br>

</div>

- [Abstract](#abstract)
- [Install & Usage](#install--usage)
- [Additional information](#additional-information)

## Abstract

Accurately measuring the sound field inside of a room purely from real-world data is extremely time-consuming, as it requires a large number of measurements and expensive equipment. In the past decade, several purely physics and signal-processing-based techniques have been proposed to predict the sound field inside a region from a limited number of data points. More recently, data-driven methods have also been proven to be powerful tools for this task. In this work, SFR3 is proposed, a sound field reconstruction diffusion model leveraging Super-Resolution via Repeated Refinement (SR3), which reconstructs the real and imaginary parts of the sound field from a grid of uniformly spaced microphones over a certain region. The model's performance is compared to a naive baseline consisting of a bicubic interpolation, a kernel-based method commonly used for this task, and a data-driven approach, showing that the proposed method outperforms the aforementioned techniques, especially at lower frequencies.

## Install & Usage

You can create the virtual environment and install the needed packages using conda with the following command: 

```
conda env create -f environment.yml
```

## Additional information

For more details:
"[Sound Field Reconstruction via Repeated Refinement: SR3-Based Room Transfer Function Interpolation Using Image Super-Resolution](https://github.com/rickgiantsteps/sfr3)" (Riccardo Passoni, Stefano Damiano, Philip J.B. Jackson, Toon van Waterschoot, Enzo De Sena)

If you use code or comments from this work, please cite:

```BibTex

```
