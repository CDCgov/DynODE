**The SEIP Model**

### Model Description

The SEIP (Susceptible-Exposed-Infectious-Partially Immune) model presented here tracks the dynamics of disease transmission across multiple dimensions. This model incorporates various compartments to reflect the following:

- **Age Groups**: Different segments of the population categorized by age.
- **Immune History**: Previous infections or exposures.
- **Vaccination Status**: Number of vaccine doses received.
- **Waning States**: Different stages of waning immunity.
- **Infection Strains**: Multiple pathogen strains that can infect individuals.

The model also includes seasonal vaccination effects, where individuals in the highest vaccination tier are periodically reset to the next highest tier, simulating annual vaccination campaigns. The mathematical representation of the model follows:

$$
\begin{align*}
{\frac{dS_{i,j,k,m}}{dt}} &= -\big(\sum_{\ell}\lambda_{i,\ell}(t) \big)S_{i,j,k,m} - (1-\delta_{m=0}\delta_{k=K}) \min\left(\frac{\nu_{i,k}(t)\mathcal{N}_{i}}{\sum_{j,m'}S_{i,j,k,m'}}, 1\right)S_{i,j,k,m} \\
&\quad + \delta_{m=0}(1-\delta_{k=0})\min \left(\frac{\nu_{i,k}(t)\mathcal{N}_{i}}{\sum_{j,m'}S_{i,j,k,m'}}, 1\right)\sum_{m'}S_{i,j,k-1,m'} \\
&\quad + \delta_{m=0}\delta_{k=K}\min \left(\frac{\nu_{i,K}(t)\mathcal{N}_{i}}{\sum_{j,m'}S_{i,j,K,m'}}, 1\right)\sum_{m'}S_{i,j,K,m'} \\
&\quad + \delta_{m=0}\sum_{j,\ell} \gamma_{\ell} I_{i,\eta(j,\ell),k,\ell} +\phi(t)(\delta_{k=K-1}S_{i,j,K,m}-\delta_{k=K}S_{i,j,K,m}) \\
&\quad + (1-\delta_{m=0})\omega_{m-1} S_{i,j,k,m-1} - (1-\delta_{m=M})\omega_{m} S_{i,j,k,m} \\
\frac{dE_{i,j,k,\ell}}{dt} &= \lambda_{i,\ell}(t)\sum_{m}S_{i,j,k,m} - \sigma_{\ell} E_{i,j,k,\ell} + \phi(t)(\delta_{k=K-1}E_{i,j,K,m}-\delta_{k=K}E_{i,j,K,m}) \\
\frac{dI_{i,j,k,\ell}}{dt} &= \sigma_{\ell} E_{i,j,k,\ell} - \gamma_{\ell} I_{i,j,k,\ell} + \phi(t)(\delta_{k=K-1}I_{i,j,K,m}-\delta_{k=K}I_{i,j,K,m}) \\
\frac{dC_{i,j,k,\ell}}{dt} &= \lambda_{i,\ell}(t)\sum_{m}S_{i,j,k,m}
\end{align*}
$$

$$
\phi(t) = 
\begin{cases}
\left(\sin\left(\frac{2\pi (t + \tau)}{730}\right)\right)^{1000} & \text{during seasonal vaccination periods} \\
0 & \text{otherwise}
\end{cases}
$$

$$
\tau = 182.5 - \Delta_{t}
$$

### Variables and Parameters

<details>
<summary>Variables and Parameters</summary>

| Variable/Parameter       | Description |
|--------------------------|-------------|
| $S_{i,j,k,m}$            | Number of individuals in age group $i$, with immune history $j$, currently in waning compartment $m$ for vaccination history $k$. |
| $E_{i,j,k,\ell}$         | Number of exposed individuals in age group $i$, with immune history $j$, vaccination history $k$ and for strain $\ell$. |
| $I_{i,j,k,\ell}$         | Number of exposed individuals in age group $i$, with immune history $j$, vaccination history $k$ and for strain $\ell$. |
| $C_{i,j,k,\ell}$         | Number of exposed individuals in age group $i$, with immune history $j$, vaccination history $k$ and for strain $\ell$. |
| $\lambda_{i,\ell}(t)$    | Force of infection, a time dependent differentiable rate at which susceptible individuals in age group $i$ become exposed to strain $\ell$. |
| $\beta_{\ell}$           | Transmission rate for strain $\ell$. |
| $\sigma_{\ell}$          | Rate at which exposed individuals for strain $\ell$ become infectious. |
| $\gamma_{\ell}$          | Rate at which infectious individuals for strain $\ell$ recover. |
| $\nu_{i,k}(t)$           | Vaccination rate, time dependent piecewise differentiable rate, for each age group $i$ and vaccination count $k.$ |
| $\omega_m$               | Waning rate for waning state $m$. |
| $\phi(t)$                | Seasonal vaccination effect modifier, continuously differentiable time dependent function. |
| $\tau$                   | Adjustment for seasonal vaccination effect timing. |
| $\Delta_{t}$             | Number of days between the start of the simulation and the date when the vaccination season changes. |
| $\eta(j, \ell)$          | Determines the new immune state given the current immune history and the exposing strain. |
| $\delta$                 | Kronecker Delta. |
| $K$                      | Maximum number of vaccination count. |
| $\mathcal{N}_{i}$        | Total population age stratification. |

</details>