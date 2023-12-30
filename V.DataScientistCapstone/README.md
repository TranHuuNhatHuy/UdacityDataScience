# DATA ANALYSIS - SIGNIFICANT EARTHQUAKES, 1965-2016

## Contents

I live in Tokyo. Recently there are not much earthquakes around here, and to be honest I quite miss that thrilling experience of feeling the vibrant through my body. Coincidentally, I am taking Udacity Nanodegree - Data Scientists. Thus, I would like to have a dive into earthquake data, with these 3 questions:

1. Which places are most vulnerable to earthquakes and seismic activities around the world, and in Japan?
2. What are the nature characteristics of heavy earthquakes? Which factors determine a strong earthquake?
3. Sometimes a seismic activity is caused by nuclear explosion. How can we detect a nuclear explosion based on seismic data solely?

The complete blog of this analysis could be viewed on my [Medium blog](https://medium.com/@trnhunhthuy).

## Project structure

- `data_analysis.ipynb` : Python notebook containing the full data analysis process.
- `data/database.csv` : CSV file containing the earthquake dataset used for this project.
- `requirements.txt` : Python package configuration file.
- `assets/*` : directory containing all figures and images used in Python notebook and in Medium blog.

## Installation

`pip install -r requirements.txt`

## Libraries used

The following Python libraries are used in this analysis:
- numpy == 1.22.1
- pandas == 1.4.2
- matplotlib == 3.5.1
- seaborn == 0.11.2
- geopandas == 0.14.0

## Licensing, Authors, and Acknowledgements

- Data source: [Significant Earthquakes, 1965-2016](https://www.kaggle.com/datasets/usgs/earthquake-database)
- Author: [US Geological Survey](https://www.usgs.gov/)
- License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)