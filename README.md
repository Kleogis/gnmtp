# Generate Novel Molecules With Target Properties Using Conditional Generative Models
Code for the paper `Generate Novel Molecules With Target Properties Using Conditional Generative Models`.

Link to [Paper](https://abhinavsagar.github.io/files/gnmtp.pdf).

## Abstract

Drug discovery using deep learning has attracted a lot of attention of late as it
has obvious advantages like higher efficiency, less manual guessing and faster
process time. In this paper, we present a novel neural network for generating
small molecules similar to the ones in the training set. Our network consists of
an encoder made up of bi-GRU layers for converting the input samples to a latent
space, predictor for enhancing the capability of encoder made up of 1D-CNN
layers and a decoder comprised of uni-GRU layers for reconstructing the samples
from the latent space representation. Condition vector in latent space is used for
generating molecules with the desired properties. We present the loss functions
used for training our network, experimental details and property prediction metrics.
Our network outperforms previous methods using Molecular weight, LogP and
Quantitative Estimation of Drug-likeness as the evaluation metrics.

## Data

The dataset can be downloaded from [here](http://zinc.docking.org/).

A sample of 100,0000 SMILES strings of drug like molecules was randomly sampled from ZINC
database. We use 90,000 molecules for training and 10,000 molecules for testing the property
prediction performance. A special sequence indicating end of sequence is appended at the end of
every sequence. To evaluate the performance of our network, we used three properties molecular
weight (MolWt), Wildman Crippen partition coefficient (LogP) and quantitative estimation of druglikeness (QED).

## Network Architecture

![results](images/d5.png)

## Usage

Calculate molecular properties:

`python calculate.py --input_filename=smiles.txt --output_filename=output.txt`

Train model:

`python train.py --prop_file=output.txt --save_dir=./save`

Generate molecules with desired properties (Example MW=300, LogP=4, and TPSA=100):

`python sample.py --prop_file=output.txt --save_file=save/model.ckpt --target_prop='300 4 100' --result_filename=result.txt`

## Benchmarks

The following benchmarks was used to determine the performance of our network for generating
molecules:

1. `Validity`: It assesses whether the molecules generated are realistic or not. Examples of not valid
molecules are one with wrong valency configuration or wrong SMILES syntax.

2. `Uniqueness`: It assesses whether the molecules generated are different from one another or not.

3. `Novelty`: It assesses whether the molecules generated are different from the ones in the training
set or not.

## Property Prediction

An important tool for evaluating the performance of network is done using properties distribution.
The following three properties are used:

1. `Molecular weight (MW)`: It is the sum of atomic weights in a molecule. To figure out if the
generated samples are biased towards lighter or heavier molecules histograms of molecular weight
for the generated and test sets are plotted.

2. `LogP`: It is the ratio of a chemical’s concentration in the octanol phase to its concentration in the
aqueous phase.

3. `Quantitative Estimation of Drug-likeness (QED)`: It is a measure of how likely a molecule is a
viable candidate for a drug. It’s value lies between 0 and 1 both included.

## Results

### A few randomly selected, generated molecules. Sampled molecules using different signature and standard deviation of 0.05:

![results](images/d1.png)

### Molecules generated by our network with the condition vector made of the five target properties of Aspirin:

![results](images/d2.png)

### Molecules generated by our network with the condition vector made of the five target properties of Tamiflu:

![results](images/d3.png)

### Generated samples using interpolation:

![results](images/d6.png)

## Comparison with SOTA

![results](images/d7.png)

## License

```
MIT License

Copyright (c) 2020 Abhinav Sagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

