---
title: "Time Series Analysis"
excerpt: "AR model, MA model, ARIMA model, Dynamic Regression.  "
categories:
  - Statistics




---



# Time Series Decomposition

## Problem Definition

Given a time series: $ y = S + T + E $

Decompose into:
- **T**: Trend/Cycle
- **S**: Seasonal
- **E**: Error (White Noise)


## Extract Trend
- Moving Average
    * For example: 7 \* MA for weekly data
    
- Moving Average of Moving Average 
    * For example: 2 \* 12 MA for monthly data, 2 * 4 MA for quarterly data
    * In R: `ma(time_series, order=12, centre = TRUE)` 2 \* 12 MA 
    
- Weighted Moving Average
    * 2 * 4 MA is a special case. i.e., W = [1/8, 1/4, 1/4, 1/4, 1/8] 

## Decomposition
Basic Approach:
1. Calculate De-trended data by using MA or 2 - m MA or m MA: **T**
1. Simple average of, for example, all January data. Adjust 12 values to sum up to zero. **S**
1. The remainder is error **E**

---


Issues: 
- No observation for beginning/ending
- Constant seasonal components over years
- Not robust to outliers

---


Other Methods:
- X-12-ARIMA Decomposition 
- STL Decomposition
    * Handle any type of seasonality
    * Change of seasonality over time
    * Users have control over smoothness
    * Robust to outliers
    
---

Forecast with decompositions:
- Naive forecast for seasonal component (assume no change, take from last year)
- For T and E
    * Random walk with drift model
    * Holt's method
    * non-seasonal ARIMA with differencing

# Time series forecasting

<img src="../assets/figures/ExponentialSmoothing.png" width="600">

Ref: 
  - https://robjhyndman.com/talks/MelbourneRUG.pdf

## Simple Exponential Smoothing


- $\hat x_{t+1} = \alpha x_t + (1-\alpha) \hat x_{t\vert t-1}$
- $\hat x_{t+1} = \hat x_{t\vert t-1} + \alpha (x_t - \hat x_{t\vert t-1})$
- $F_{t+1} = F_{t} + \alpha (A_{t} - F_{t})$
- $F_{t+1} = \alpha A_t + (1 - \alpha) F_t$

## Holt-Winters Additive method
- Main idea
    * Base
    * Error
- Key:
    - Y := L + 1 * b + S
    - L := L + b
    - S := S
    - b := b
    
    - L1 = (Y - S) 
    - L2 = L + b
    
    - S1 = Y - l - b
    - S2 = S
    
    - b1: L - L
    - b2: b

- Forecast = **level** + **trend** + **seasonal component** <br/>
$ \hat{y}_{t+h\vert t} = l_t + hb_t + s_{last} $


- Level = Seasonal Adjusted Observation + Non-seasonal Forecast for t <br/>
$l_t = \alpha(y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + b_ {t-1})$


- Trend = Change in level + Trend from last year <br/>
$ b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}$


- Seasonal = Current seasonal index + Seasonal index from last year <br/>
$ s_t = \gamma(y_t - l_{t-1} - b_{t-1} )+ (1-\gamma)(s_{t-m}) $


- Unified error correction form:<br/>
$ \theta := \theta + \alpha * error $<br/>
$ error = y_t - (l_{t-1} + b_{t-1} + s_{t-m}) $

## Other methods
1. Damped Trend Model:
    - Short-run: trended
    - Long-run: constant
    - $ \hat{y}_{t+h\vert t} = ... + (\phi + \phi + ... + \phi^h)b_t + ... $
1. Exponential Trend Model
    - $ \hat{y}_{t+h\vert t} = l_tb^h_t $
1. Holt's linear Trend model
    - No seasonal term




# AR(1) and MA(1) Model
## Defniition of stationary series
**Definition of weak stationarity**

- $ E(Y_t) = 0 $ <br/>
- $ Var(Y_t) = constant $
- $ Cov(Y_t, Y_{t-k}) = \gamma_k $
- Think of The covariance matrix

## AR(1) Model
$$ Y_t = c + \phi Y_{t-1} + \epsilon_t,\ where\ \epsilon\ - iid(0, \sigma^2) $$



- Mean: 
    - $Y_t = c \sum_{i=0}^{t-1}\phi^i + \phi^t Y_0 + \sum_{i=0}^{t-1} \phi^i a_{t-i}$
    - $E(Y_t) = c \sum_{i=0}^{t-1}\phi^i + \phi^t Y_0$


- Condition for stationary
    - When $\vert \phi\vert <1,$ $\mu = E(Y_t) = \frac{c}{1-\phi}$
    - Root of operator: $(1-\phi B) = 0, B= \frac{1}{\phi}$
    - if $ \phi = 1 $ and $ c = 0 $ : random walk
    - if $ \phi = 1 $ and $ c <> 0 $ : random walk **with drift**


- Variance
    - If c=0, $Y_t - \phi Y_{t-1} = (1-\phi B) = \epsilon_t$
    - If c=0, $\sigma^2_Y = \phi^2 \sigma^2_Y + \sigma^2_{\epsilon}$, $\sigma^2_Y = \frac{\sigma^2_{\epsilon} }{1- \phi^2}$
    
- Autocovariance
    - If c=0, $\gamma_k = E(Y_{t-k}Y_t) = E[Y_{t-k}(\phi Y_{t-1} + \epsilon_t)] = \phi  E(Y_{t-k}Y_{t-1}) = \phi \gamma_{k-1}$
    - $\gamma_0 = E(Y_{t}Y_t) = \sigma^2_Y$
    
- Representation by error term
    - If c=0, $Y_t = \epsilon_t + \phi \epsilon_{t-1} + \phi^2 \epsilon_{t-1} + ...$
    - $Y_t = \sum_{j=0}{}{\phi^j \epsilon_{t-j} }$, which is $MA(\infty$) with special structure for weights
    - Indication: **keep "long" memory with decreasing weights**


## MA(1) Model
$ Y_t = c + \epsilon_t - \theta\epsilon_{t-1},\ where\ \epsilon\ - iid(0, \sigma^2) $
<br/>
- If c=0, $Y_t = (1-\theta B)\epsilon_t$
- Always stationary


- Mean
    - $E(Y_t) = \mu$
    
- Variance
    - $\sigma^2_Y = E(Y^2_t) = \sigma^2_{\epsilon}(1+\theta^2)$
    
- Covariance
    - $\gamma_1 = E(Y_t Y_{t-1}) = -\theta \sigma_{\epsilon}^2$
    - $\gamma_2 = \gamma_3 = ... = 0$

Indication: **noise / shock quickly vanishes with time.**

Note: Difference between MA *model* and MA *smoothing*
- MA model: forecast stationary series
- MA *smoothing*: forecast trend


## Comparison/ Connection between AR and MA

- AR model can be represented by $MA(\infty)$ model with restrictions on the decay pattern of coefficients

- MA model has finite terms with no restrictions on coefficients

- AR model has many non-zero autocorrelation with decay pattern

- MA model has a few non-zero autocorrelation with no restriction

- It can be proved that:
    - $AR(p) + AR(0) = ARMA(p,p)$
    - $AR(p) + AR(q) = ARMA(p + q,max(p,q))$
    - $MA(p) + MA(q) = MA(max(p,q))$

# ARIMA model 

## Integrated Process / Non-stationary

- I(2) means the series need to be differenced TWICE in order to be stationary
- For example: random walk: $ Y_t = Y_{t-1} + \epsilon_t $ is $I(1)$
- For example: stationary process: $I(0)$


- Special Case : $Y_t - Y_{t-1} = c + (\epsilon_t - \theta \epsilon_{t-1})$
- $\theta=0$: random walk
- $c=0, \vert \theta\vert  <1$: simple exponential smoothing

***Random walk***

$$ Y_t = c + \phi Y_{t-1} + \epsilon_t$$

- If $\phi = 1$, $\Delta Y_t = c + \epsilon_t$
    - Or: $Y_t = ct + \epsilon_{t} + \epsilon_{t-1}+ \epsilon_{t-2} + ...$
- Unlike stationary process, constant $c$ is very important in defining non-stationary process
    - $E(Y_t) = ct $ 
    - $\sigma^2_Y = \sigma^2_{\epsilon}t$
    - $cov(t, t+k) = \sigma^2_{\epsilon}t$
<img src="../assets/figures/random_walk.png" width="300">

***Simple Exponential Smooth (SES)***

* $ y_t = Y_t - Y_{t-1} = \mu - \theta \epsilon_{t-1} + \epsilon_t$, where it is a combination of *deterministic trend* and *stochastic trend*.
* $ \mu$ is the constant term. Let $\mu=0, \vert \phi\vert<1$,
    - $E(Y_t) = \mu t$. If $\mu$ = 0, $ Y_t - Y_{t-1} =  - \theta \epsilon_{t-1} + \epsilon_t$.
    
* $ Y_t = \epsilon_t + Y_{t-1} - \theta\epsilon_{t-1} = \epsilon_t + Y_{t-1} - \theta(Y_{t-1} - Y_{t-2} + \theta\epsilon_{t-2}) + ...... $
* $ Y_t = \epsilon_t + (1-\theta) Y_{t-1}  + \theta(1-\theta)Y_{t-2} + \theta^2(1-\theta) Y_{t-3} +......$
* Equivalent: $AR(\infty)$ with infinite geometric progression

## Seasonality

- Base: $ y_t = \mu + \phi_1 y_{t-1} + ... + \phi_p y _{t-p} + \theta_1 e_{t-1} + ... + \theta_q e_{t-q} + e_t $

- Seasonal Differencing
    - Seasonaility: $E(Y_t) = E(Y_{t-s})$ where $Y$ is de-trended. The series has a seasonal period of $s$


- Types of Seasonality
    - let $n_t$ to be stationary, then $Y_t = S_t^{(s)} + n_t$
    - Deterministic effect:  $ S_t^{(s)} = S_{t+s}^{(s)} = S_{t+2s}^{(s)} = S_{t+3s}^{(s)} = ......$
    - Stationary effect: $ S_t^{(s)} = \mu^{(s)} + v_t$, where $\mu^{(s)}$ is mean for each season, and $v_t$ is another stationary process
    - Non-stationary effect: $ S_t^{(s)} = S_{t-s}^{(s)} + v_t$
    - Note: **Seasonal MA and AR terms**


- Seasonal differencing    
    - Convert non-stationary with seasonality to stationary process
    - Example: **$ARIMA(1,1,1)(1,1,1)_4$ without constant**
<img src="https://i.stack.imgur.com/NUA6V.png" width="500">

## Model Identification
### Test for stationarity

**Dickey Fuller Test of Stationarity (for AR1)**

- $ Y_t = \phi Y_{t-1} + \epsilon_t$
- $Y_t - Y_{t-1} = (\rho - 1) Y_{t-1} + \epsilon_t$
- Intuition: higher value will be followed with a decrease, and lower value will be followed with an increase; 
- Random walk with $\phi$ = 1 is not stationary since the last position do not imply increase or decrease
- Test if $(\rho-1)$ is zero or not, i.e., if $\rho$ is equal to one; If zero, then non-stationary

**Augmented Dickey Fuller (ADF) Test of Stationarity (for ARMA)**

- $Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \phi_3 Y_{t-3} +... +\epsilon_t $
- $Y_t - Y_{t-1} = \rho Y_{t-1}  - \alpha_1 (Y_{t-1} - Y_{t-2}) - \alpha_2 (Y_{t-2} - Y_{t-3}) - ... + \epsilon_t $
- Intuition: for a non-stationary series, $Y_{t-1}$ will not provide relevant information in predicting the change in $Y_t$ besides the lagged changes $\Delta$
- In other words: measure if the contribution of lagged value $Y_{t-1}$ is significant or not
- How to lag length `k`? Use AIC, BIC for model selection, or default $(T-1)^{1/3}$

**Variations**
- Other options: KPSS test, hypothesis opsite


### Transformations

- Variance stablizing
    - Log
    - Square root
    - Box-cox transformation


- Mean stablizing
    - Regular differencing
    - Seasonal difference
    
- Log: fix exponentially trending
- Detrend: Y = (mean + trend * t) + error; Model trend from here 
- Differencing: 
    * First-order differencing: $Y_t - Y_{t-1} = ARMA(p,q)$ 
    * Seasonal differencing with period m: $Y_t - Y_{t-m} = ARMA(p,q)$
    * Here the order of differencing is `I` in AR(I)MA

### Identify `p` and `q`

**Two useful graphs**
- Auto Correlation Function (ACF):
    * A lag k aurocorrelation: $Corr(Y_t, Y_{t-k})$
    * AR(1): Gradually decrease with lag k
    * MA(1): Spike at lag 1, then zero for lag k > 1
    
- Partial Correlation Function (PACF):
    * Only measure the association between $Y_t, Y_{t-k}$
    * Exclude the effect of $Y_{t-1}, ..., Y_{t-(k-1)} $
    * $Y_t = \beta_1 Y_{t-1} + \beta_2 Y_{t-2} + u_t$
    * $Y_{t-3} = \gamma_1 Y_{t-1} + \gamma_2 Y_{t-2} + v_t$
    * $PACF(t, t-3) = corr(u_t, v_t)$
    

<img src="../assets/figures/acf_pacf.png" width="500">

---

<img src="../assets/figures/acf_pacf_2.png" width="500">

---

<img src="../assets/figures/acf_pacf_3.png" width="500">



## Model estimation and selection
- Use repeated KPSS tests to determine differenced d to achieve stationary series
- Use *Maximum Likelihood Estimation* to minimize $ e^2_t $
- The value of `p` and `q` are selected by minimizing $AIC$ using some search strategy
    - $AIC = -2log(L) + 2K = Tln \hat\sigma^2_{\epsilon} + 2K$
    - Error + Number of parameters
- Start from base ARIMA and add variations until no lower $AIC$ found
<img src="../assets/figures/aic_bic.png" width="500">



## Model diagnostics for residuals
- Zero mean
- Constant variance
- No autocorrelation
- Normal distribution

## Forecasting and Evaluation
- https://www.otexts.org/fpp/8/8
<img src="https://i.stack.imgur.com/83BUy.png" width="400">




# Dynamic Regression: 


## ADL model (Autoregressive Distributed Lag) Model

- Formulation: $Y_t = \alpha + \delta t + \phi_1Y_{t-1} + \phi_2Y_{t-2} + ... + \phi_p Y_{t-p} + \beta_0X_t + \beta_1X_{t-1} + ... + \beta_qX_{t-q} + \epsilon_t$
- Where $\epsilon_t$ ~ $iid(0, \sigma^2)$

### If X and Y are stationary `I(0)`
- Run OLS on ADL model
- For interpretation purpose: rewrite ADL to be $\Delta Y_t = \alpha + \delta t + \phi Y_{t-1} + \gamma \Delta Y_{t-1} +\gamma_2\Delta  Y_{t-2} + ... + \gamma_p\Delta  Y_{t-p} + \theta X_t + \omega_1\Delta X_{t-1} + ... + \omega_q\Delta X_{t-q} + \epsilon_t$


- Long Term effect: $Y-Y = \phi Y + \theta X$
    * $\partial Y / \partial X = -\theta / \phi$
    * If X permanently increase by 1%, what percent with Y change
    
- Short Term effect: Not clear

### If X and Y are I(1)

In other words, X and Y has unit root I(1)

$$Y_t = \alpha + \beta X_t + e_t$$

- **Spurious Regression**: 

    - $\beta$ should be zero, but estimated $\beta$ not zero; ($e_t$ has a unit root, e_t is not stationary)
    - In other words, estimation is biased
    - Cannot use t tests because distributionb is no longer **t** or normal (error structure)
    - Possible fix: $\Delta Y_t = \Delta X_t + \Delta e_t$ as long as e is I(1), then $\Delta e_t$ is stationary. But different interpretation.
    
- **Cointegration**:
    -$e_t$ does **NOT** has a unit root --> $e_t$ is stationary, and is called **equilibrium error**
    - Premise: there exist unit root for X and Y
    - Test method (Engle-Granger Test): run unit root test (Dickey-Fuller Test) on residual $\hat Y_t - \hat\alpha - \hat\beta X_t $
    
- Under coitergration: still can run OLS with Y on X (**Cointegrating Regression**)
    - OLS: Estimate $Y_t = \alpha + \beta X_t $
    - $\beta$ is super-consistent
    - T stats not interpretable

    
    
- Under coiteration: call also run full ADL

### Error Correction Model (ECM)

- Premise: X and Y cointegrating I(1)
- Long-Run OLS: Estimate $Y_t = \alpha + \beta X_t + e_t$
- Short-Run OLS: $\Delta Y_t = \gamma \hat e_{t-1} + \omega_0 \Delta X_t + \epsilon_t$ 
- The short-run OLS above applies for **AR(1)**, can be easily proved. More lags for y and x can be added for arima models.
- Where $\hat e_t = Y_{t-1} - \hat \alpha - \hat \beta X_{t-1}$ and $\gamma <0$
- $\omega$ is the short-term effect  from $\Delta X$
- $e_{t-1}$ is error correction term, move towards equilibrium

Relationship with ADL:
- Special Case of ADL for I(1) variables

Ref: http://web.sgh.waw.pl/~atoroj/econometric_methods/lecture_6_ecm.pdf

## A special case: AR(1) for error term
- **Example of AR(1)**: Formulation
    * $y_t = \alpha + \beta x_t + \epsilon_t,\ where\ errors\ (\epsilon_t)\ is\ autocorrelated$ <br> 
        * What happens: solution not efficient any more, and statistical tests no longer apply
    * $ Assume\ \epsilon_t = \rho \epsilon_{t-1} + \omega_t\ where\ \omega\ - iid(0, \sigma^2) $ <br/>
    * Note: More appropriate ARMA model can be available
    * Similary, it can be shown/proved that the series $ \epsilon_t $ is stationary


- Rewrite assumtpion for stationary error
    * $ E(\epsilon ) = 0 $ <br/>
    * $ E(\epsilon^2 \vert X ) = \rho(\frac{\sigma^2}{1-\rho^2}) $ Homescedasticity <br/>
    * $ E(\epsilon_i \epsilon_j) = \rho_{\vert i-j\vert } * \sigma^2 $ What matters is proximity $k = \vert i-j\vert $
        * $ Corr(\epsilon_t, \epsilon_{t-1}) = \rho $
    
- Model assumptions
    * Stationarity for Y and X
    * Differencing may be needed
    
- How to solve for $\beta$?
    * One way: **Cochrane-Orcutt Method (Yule-Walker Method)**
    * OLS: $\hat{\epsilon_t} = y_t - \hat{\alpha} - \hat{\beta} * x_t$
    * OLS: $\hat{\epsilon_t} = \rho \hat{\epsilon_{t-1} } + \omega_t,\ solve\ for\ \hat{\rho}$
    * Re-formulate: $y_t^* = t_t - \rho y_{t-1} = \alpha(1-\hat{\rho}) + \beta* x_t^* + \omega_t,\ solve\ for\ \hat{\alpha}, \hat{\beta} $
    * Where $ y_t^* = t_t - \hat\rho y_{t-1}$
    * Re-iterate until convergence
    
    
    
- How to predict?
    * $F_{t+1} = \hat{y}_{t+1} + re_t$, combining regression part and ARMA part
    * how about X? model separately, given, or assume future values

## Test for auto-correlation of residuals
**Durbin-Watson test**
- $\epsilon_t = \rho \epsilon_{t-1} + \omega_t\ $
- Hypothesis
    * $H_0: \rho = 0$<br/>
    * $H_1: \rho <> 0$

- Test statistics

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7ldvxO8eWwkHL64DbkWEPQFkp8oMrRWEZDqjAcnLafBgJhjf6" width="200">

**Ljung-Box Q Test**
- Hypothesis
    * $H_0$: the autocorrelations up to lag k are all zero
    * $H_1$: At least one is not zero

- Test statistics

<img src="http://file.scirp.org/Html/2-1630023/1a6d69eb-051a-4f24-978c-ff1fd0cb5071.jpg" width="200">


