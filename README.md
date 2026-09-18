# Multivariate LSTM Modeling for Wastewater Quality Prediction
Innocentia Bénédicta Sossou, Sophie Laruelle, Jacques Printems
Invalid Date

[![build and
publish](https://github.com/computorg/template-computo-r/actions/workflows/build.yml/badge.svg)](https://github.com/computorg/template-computo-r/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

### Authors

- [Innocentia Bénédicta
  Sossou](https://sites.google.com/view/innocentia-sossou-en/home) (Univ
  Paris Est Creteil, Univ Gustave Eiffel, CNRS, LAMA UMR8050, F-94010
  Creteil, France)
- [Sophie Laruelle](https://perso.math.u-pem.fr/laruelle.sophie/)
  (**?meta:by-affiliation.2.name**)
- [Jacques Printems](https://perso.math.u-pem.fr/printems.jacques/)
  (**?meta:by-affiliation.3.name**)

### Abstract

The proposed approach relies on high-frequency multivariate time series
collected at the Seine aval (SIAAP ) treatment plant, including pH,
temperature, conductivity, and total suspended solids, as well as
exogenous variables related to precipitation and measurements from an
upstream plant. The objective is to forecast wastewater quality over a
24-hour horizon. A multivariate Long Short-Term Memory (LSTM) recurrent
neural network is implemented to capture complex temporal dependencies
and nonlinear patterns in the data. The model is trained on one year of
data and validated on four months of data (one per season), and is
compared with a persistence model and SARIMA models. The evaluation
shows an overall superiority of the LSTM model, particularly for
variables exhibiting high levels of noise and nonlinearity.
