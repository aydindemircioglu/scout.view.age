
# Pediatric Age Estimation from Thoracic and Abdominal CT Scout Views using Deep Learning

This is the code repository for the paper
"Pediatric Age Estimation from Thoracic and Abdominal CT Scout Views
using Deep Learning".


## Requirements

All coding was done using Python 3.8.10. Needed modules can be found in
requirements.txt.


## Images

Because of privacy, no images and no code for preparing these have been
uploaded. Images should be put into ./images; corresponding data frames
containing the image path and the ages should be put into ./data.
The dataset.py file must be adapted correspondingly.


## Training

Five different methods were tested: Simple regression using L1 loss, regression
using L1 loss but also DICOM tags, and three more sophisticated methods, AMR,
CORAL and CORN. For each of these methods, a hyperparameter tuning was performed
using Optuna 3.0.
To start tuning, execute ./train_optuna.py in each of the subdirectories.
This will create a CSV with the performance of each tested network.


## Testing

Afterwards the best model is retrained on all data by using ./retrain.py
Actually this method first reproduced the best model (to check whether everything
works and is reproducable) and then retrains and evaluates on all data.
Only the evaluation of the best-performing model is considered-- I did not
care about the MAE of the other models, but out of comfort, still train and
evaluate them.


## Evaluation

For evaluation, call the ./evaluate.py script. For the visualization,
execute ./visualize.py. For the equivalence test, you will need R
and the TOSTER library. The script is eqTest.R.


## Results

Results of the optuna run and the final predictions of all models on the
validation and test set can be found in ./results

In addition, the pngs of all generated figures can be found in ./paper.



## LICENCE

Copyright 2022, Aydin Demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
