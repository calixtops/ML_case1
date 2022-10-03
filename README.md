# Case Machine Learning
## Project: Estimation of billing and potential for neighborhoods in the city of São Paulo using information from the city of Rio de Janeiro

### Case description 

A food company located in Rio de Janeiro wants to open branches in the city of São Paulo, targeting adults aged 25 to 50, from classes A (A1 and A2 incomes) and B (B1 and B2 incomes).

This company already operates in Rio de Janeiro and before making the investment wants to estimate how much it could make in each of the São Paulo neighborhoods as well.

Since the São Paulo market is not well known to this company, it wants to classify the São Paulo neighborhoods to know if the potential is high, medium or low. Using as a base the potential that they already know of the neighborhoods in Rio de Janeiro. The analysts of this company would also like to segment the neighborhoods of São Paulo according to income and age profile to target differentiated marketing actions. differentiated marketing actions.

The Data Engineer helped extract the billing and potential data potential of the neighborhoods of Rio de Janeiro from our client, together with the socio-demographic data of the neighborhoods in Rio de Janeiro and São Paulo.

The attached CSV has this data for you to perform your your analysis.

Now, put together an analysis of the São Paulo neighborhoods to:
- Estimate the revenue that a store would have in each of the neighborhoods
- Classify the potential of each neighborhood as High, Medium or Low
- Segment the neighborhoods of São Paulo according to income and age, and
age, and indicate those with greater adherence to the target public.

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 


### Code

In the 'src' folder, we have the following routines:

- regression_classification.py

It runs the regression and classification algorithms for the neighborhoods of São Paulo. Read the data, preprocesses the dataset, and creates a new dataset with the expected results for the São Paulo neighborhoods.

- clustering.py

Segments the neighborhoods of São Paulo using Agglomerative Clustering.

- plots.py

Makes the plots that will be used in the presentation to the client.

- utils.py

Library with the functions defined for the project execution. 

ex: 

```sh
import utils as gf
```


## Instale as bibliotecas necessarias

Install the required libraries

```sh
pip install -r src/requirements.txt
```


## Run

In the root folder of the project, run:

#### To run the analysis:
```sh
python src/regression_classification.py
```

#### To run the plots:
```sh
python src/plots.py
```

## Outputs

The output .csv file with the regression will be saved in the "data" folder

The .png plots will be saved in the "figures" folder

## Dashboard

The dashboard for viewing the results is in the project [dash_ML_case1](https://github.com/calixtops/dash_ML_case1)
