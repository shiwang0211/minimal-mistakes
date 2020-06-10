# Prepare Data


```R
set.seed(5)
N = 100
data = data.frame(x1 = seq(1,N,1), x2 = log(seq(1, N, 1)))
data$y = 3 * data$x1 + 4 * data$x2 + rnorm(N, 0, 1) + 5
rownames(data) = seq(ISOdate(2000,1,1), by = "month", length.out = N)
head(data)
```


<table>
<thead><tr><th></th><th scope=col>x1</th><th scope=col>x2</th><th scope=col>y</th></tr></thead>
<tbody>
	<tr><th scope=row>2000-01-01 12:00:00</th><td>1        </td><td>0.0000000</td><td> 7.159145</td></tr>
	<tr><th scope=row>2000-02-01 12:00:00</th><td>2        </td><td>0.6931472</td><td>15.156948</td></tr>
	<tr><th scope=row>2000-03-01 12:00:00</th><td>3        </td><td>1.0986123</td><td>17.138957</td></tr>
	<tr><th scope=row>2000-04-01 12:00:00</th><td>4        </td><td>1.3862944</td><td>22.615320</td></tr>
	<tr><th scope=row>2000-05-01 12:00:00</th><td>5        </td><td>1.6094379</td><td>28.149193</td></tr>
	<tr><th scope=row>2000-06-01 12:00:00</th><td>6        </td><td>1.7917595</td><td>29.564130</td></tr>
</tbody>
</table>



# Run model with `lm`


```R
model = lm("y ~ x1 + x2 + 1", data)
model
```


​    
    Call:
    lm(formula = "y ~ x1 + x2 + 1", data = data)
    
    Coefficients:
    (Intercept)           x1           x2  
          4.758        2.996        4.126  



# Likelihood Test

- Used for testing **nested** model
    - Null model: all p = 0 except intercept
    - Alternative model: OLS
    
- Assume normal distribution of error term: $L(\theta) = (\frac{1}{\sqrt{2\pi\sigma^2} })^N exp(-\frac{1}{2\sigma^2}R)$
     * $where\ R= \sum_{i=1}^{N}\hat e^2_i $
     * $\sigma$ unknown, $ \hat {\sigma}^2 = R / N$


- So, $L(\theta) = (\frac{1}{\sqrt{2\pi\sigma^2} })^N exp(-\frac{N}{2})$

- So, $Log(L(\theta)) = -\frac{N}{2} [log(2\pi + ln(R) + 1)]$


- Likelihood Ratio (LR) = $\Lambda = 2 ln (L_U / L_R) = 2[ln(L_U) - ln(L_R)] = 2ln(\frac{\sigma^2_1}{\sigma^2_0})^{-N/2}= 2ln(\frac{R_U}{R_R})^{-N/2} $

- $\Lambda$ ~ $\chi_p$. If $\Lambda$ > threshold, then alternative model is better by adding complexity


```R
# Restricted, estimated value is just avg(y)
R_0 = sum((data$y - mean(data$y))^2)
R_0
```


808675.150518816



```R
# UnRestricted model with fitted parameters
R_1 = sum(model$residuals^2)
R_1 
```


88.1948161842128



```R
# Likelihood ratio
LR = 2*log((R_1/R_0)^(-N/2))
LR
```


912.360438306004


<img src = "https://i.stack.imgur.com/PbEqv.jpg" width = 500>


```R
pchisq(LR, df=2, lower.tail=FALSE)
```


7.6462425161496e-199



```R
#Unrestrcited
loglik_1 = -N/2 * (log(2*pi) +log(R_1/N)+1)
loglik_1
```


-135.612753414258



```R
lmtest::lrtest(model)
```


<table>
<thead><tr><th scope=col>#Df</th><th scope=col>LogLik</th><th scope=col>Df</th><th scope=col>Chisq</th><th scope=col>Pr(&gt;Chisq)</th></tr></thead>
<tbody>
	<tr><td>4            </td><td>-135.6128    </td><td>NA           </td><td>      NA     </td><td>           NA</td></tr>
	<tr><td>2            </td><td>-591.7930    </td><td>-2           </td><td>912.3604     </td><td>7.646243e-199</td></tr>
</tbody>
</table>



# Granger Causality Test

$y_t = c + \sum_{i=1}^{p}\alpha_ix_{t-i} + \sum_{i=1}^{p}\beta_iy_{t-i} + \epsilon_t$

Perform F test for the null hypothesis:
$H_0: all\ \alpha_i = 0$


```R
# Create differencing series
data$y_lag1 = c(NA, data$y[c(1:(N-1))])
data$x1_lag1 = c(NA, data$x1[c(1:(N-1))])
```


```R
# Residuals for Unrestricted Model: R_u
model_u = lm("y ~ x1_lag1 + y_lag1 + 1", data)
R_u = sum(model_u$residuals^2)
R_u
```


127.697807134118



```R
# Residuals for Restricted Model: R_r
model_r = lm("y ~ y_lag1 + 1", data)
R_r = sum(model_r$residuals^2)
R_r
```


177.348529101052


$F = \frac{ (R_{restricted}-R_{unretricted}) / \Delta p }{R_{unrestricted}/ (N - P - 1)} $ ~ $F_{\Delta p, N-P-1,}$


```R
# F statistics
F = ((R_r - R_u) / 1) / ((R_u) / (N - 2 - 1))
F
```


37.7149783451981


# Linearity Test (Ramsey)

$y = \beta x + \gamma_2 \hat y^2 + ... + \gamma_k\hat y^k + \epsilon$

Perform F test on $H_0: all\ \gamma_i = 0$


```R
y_hat = model$fitted.values
data$y_2 = y_hat^2
data$y_3 = y_hat^3
data$y_4 = y_hat^4
```


```R
model_u = lm("y~x1+x2+y_2+y_3+y_4+1", data) # P = 5, Delta_P = 3
```


```R
#Residuals for Unrestricted Model: R_u
R_u = sum(model_u$residuals^2)
R_u
```


85.671236152551



```R
#Residuals for Restricted Model: R_r # 
R_r = sum(model$residuals^2)
R_r
```


88.1948161842128


$F = \frac{ (R_{restricted}-R_{unretricted}) / \Delta p }{R_{unrestricted}/ (N - P - 1)} $ ~ $F_{\Delta p, N-P-1,}$


```R
# F statistics
F = ((R_r - R_u) / 3) / ((R_u) / (N - 5 - 1))
F
```


0.922972258560666


# VIF (Variance Inflation Factor) 


```R
# 1. Regress X1 on all other X
model_vif = lm("x1~x2+1", data)
```

$R_2 = 1 - \frac{\hat e^2}{\sum_{i}{(y_i-\bar y)^2} }$


```R
# 2. Calculate R^2
R_2 = 1 - sum(model_vif$residuals^2) / sum((data$x1 - mean(data$x1))^2)
R_2
```


0.802559598696214


$VIF = \frac{1}{1-R^2}$


```R
# 2. VIF = 1/(1-R^2)
VIF = 1/(1-R_2)
VIF
```


5.06481952729309


# Turning Point Test


```R
e = model$residuals[2:(N-1)]
e_prev = model$residuals[1:(N-2)]
e_after = model$residuals[3:N]
```


```R
# calculate number of turning points
num_tp = sum((e > e_prev & e > e_after) | (e < e_prev & e < e_after))
num_tp
```


74


<img src = "https://wikimedia.org/api/rest_v1/media/math/render/svg/c1abc79ac72647ea2c419d6db5d3e33ba2520637" width = 100>


```R
z = (num_tp - (2*N-4)/3)/sqrt((16*N-29)/90)
z
```


2.07436537886071



```R
p = 2*(1- pnorm(z)) #two-tailed
p
```


0.0380453901228328


# Mann-Kendall Rank Tests

- sgn(x-y) = 1 if x>y
- sgn(x-y) = -1 if x<y
- sgn(x-y) = 0 if x==y

$S = \sum_i \sum_{j>i} sgn(x_j-x_i)$


```R
sum_sgn = 0
for (index in c(1:c(N-1))){ # index from 1 to N-1
   sum_sgn = sum_sgn + 
      sum(model$residuals[c((index+1):N)] > model$residuals[index]) -  # from (index + 1) to N
      sum(model$residuals[c((index+1):N)] < model$residuals[index])    # from (index + 1) to N
}
sum_sgn
```


142


$VAR(S) = 1/18[n(n-1)(2n+5)]$


```R
var_s = 1/18*N*(N-1)*(2*N+5)
var_s 
```


112750


$Z = \frac{S-1}{\sqrt {var(S)} }$


```R
Z = (sum_sgn - 1)/sqrt(var_s)
Z
```


0.419914467058749



```R
2 * (1 - pnorm(abs(Z), 0, 1))
```


0.674547938800206


# Cochrane-Orcutt Method (Yule-Walker Method)


* OLS: $\hat{\epsilon_t} = y_t - \hat{\alpha} - \hat{\beta} * x_t$
* OLS: $\hat{\epsilon_t} = \rho \hat{\epsilon_{t-1} } + \omega_t,\ solve\ for\ \hat{\rho}$
* Re-formulate: $y_t^* = t_t - \rho y_{t-1} = \alpha(1-\hat{\rho}) + \beta* x_t^* + \omega_t,\ solve\ for\ \hat{\alpha}, \hat{\beta} $
* Where $ y_t^* = t_t - \hat\rho y_{t-1}$
* Re-iterate until convergence

    


```R
e = model$residuals[c(2:N)]
e_lag = model$residuals[c(1:(N-1))]
e_data = data.frame(e=e, e_lag=e_lag)
error_model = lm("e~.", data=e_data)
summary(error_model)
```


​    
    Call:
    lm(formula = "e~.", data = e_data)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -2.1998 -0.6592 -0.1084  0.7271  2.4306 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept) 0.005993   0.095389   0.063    0.950
    e_lag       0.072111   0.101064   0.714    0.477
    
    Residual standard error: 0.9491 on 97 degrees of freedom
    Multiple R-squared:  0.005221,	Adjusted R-squared:  -0.005034 
    F-statistic: 0.5091 on 1 and 97 DF,  p-value: 0.4772




```R
rho= error_model$coefficients[2]
rho
```


<strong>e_lag:</strong> 0.0721113806208257



```R
y = data$y[c(2:N)] - rho * data$y[c(1:(N-1))] 
x1 = data$x1[c(2:N)] - rho * data$x1[c(1:(N-1))] 
x2 = data$x2[c(2:N)] - rho * data$x2[c(1:(N-1))] 
new_data = data.frame(y=y,x1=x1,x2=x2)
```


```R
new_model = lm("y~.", data=new_data)
summary(new_model)
```


​    
    Call:
    lm(formula = "y~.", data = new_data)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -2.19880 -0.66310 -0.08944  0.74977  2.42201 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept) 4.706244   0.676482   6.957 4.25e-10 ***
    x1          2.999333   0.009027 332.280  < 2e-16 ***
    x2          4.000856   0.304599  13.135  < 2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 0.9531 on 96 degrees of freedom
    Multiple R-squared:  0.9999,	Adjusted R-squared:  0.9999 
    F-statistic: 3.702e+05 on 2 and 96 DF,  p-value: < 2.2e-16



<img src = "http://slideplayer.com/8016538/25/images/20/Negative+Binomial+Regression.jpg" width = 300>


```R

```
