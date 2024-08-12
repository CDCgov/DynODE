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
\frac{dS_{i,j,k,m}}{dt} =
-\left(\sum_{\ell}\lambda_{i,\ell}(t)\right) S_{i,j,k,m}
- (1-\delta_{m=0}\delta_{k=K}) \min\left(\frac{\nu_{i,k}(t)\mathcal{N}_{i}}{\sum_{j,m'} S_{i,j,k,m'}}, 1\right) S_{i,j,k,m}
$$

$$
+ \delta_{m=0}(1-\delta_{k=0})\min \left(\frac{\nu_{i,k-1}(t)\mathcal{N}_{i}}{\sum_{j,m'} S_{i,j,k-1,m'}}, 1\right)\sum_{m'} S_{i,j,k-1,m'}
$$

$$
+ \delta_{m=0}\delta_{k=K} \min \left(\frac{\nu_{i,K}(t)\mathcal{N}_{i}}{\sum_{j,m'} S_{i,j,K,m'}}, 1\right)\sum_{m'} S_{i,j,K,m'}
$$

$$
+ \delta_{m=0}\sum_{{(j',\ell ') \ | \ \eta(j',\ell ')=(j,\ell)}} \gamma_{\ell} I_{i,\eta(j',\ell'),k,\ell '}
+\phi(t)(\delta_{k=K-1} S_{i,j,K,m}-\delta_{k=K} S_{i,j,K,m})
$$

$$
+ (1-\delta_{m=0})\omega_{m-1} S_{i,j,k,m-1} - (1-\delta_{m=M})\omega_{m} S_{i,j,k,m}
$$

$$
\frac{dE_{i,j,k,\ell}}{dt} =
\lambda_{i,\ell}(t) \sum_{m} S_{i,j,k,m} - \sigma_{\ell} E_{i,j,k,\ell}
+ \phi(t)(\delta_{k=K-1} E_{i,j,K,m}-\delta_{k=K} E_{i,j,K,m})
$$

$$
\frac{dI_{i,j,k,\ell}}{dt} =
\sigma_{\ell} E_{i,j,k,\ell} - \gamma_{\ell} I_{i,j,k,\ell}
+ \phi(t)(\delta_{k=K-1} I_{i,j,K,m}-\delta_{k=K} I_{i,j,K,m})
$$

$$
\frac{dC_{i,j,k,\ell}}{dt} =
\lambda_{i,\ell}(t) \sum_{m} S_{i,j,k,m}
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
| $\Lambda_{i,\ell}(t)$    | Force of infection for susceptible individuals, a time dependent differentiable rate at which susceptible individuals in age group $i$ become exposed to strain $\ell$. |
| $\beta_{\ell}$           | Transmission rate for strain $\ell$. |
| $\sigma_{\ell}$          | Rate at which exposed individuals for strain $\ell$ become infectious. |
| $\gamma_{\ell}$          | Rate at which infectious individuals for strain $\ell$ recover. |
| $\nu_{i,k}(t)$           | Vaccination rate, time dependent piecewise differentiable rate, for each age group $i$ and vaccination count $k.$ |
| $\omega_m$               | Waning rate for waning state $m$. |
| $\phi(t)$                | Seasonal vaccination effect modifier, continuously differentiable time dependent function. |
| $\tau$                   | Adjustment for seasonal vaccination effect timing. |
| $\Delta_{t}$             | Number of days between the start of the simulation and the date when the vaccination season changes. |
| $\eta(j, \ell)$          | Determines the new immune state given the current immune history $j$ and the exposing strain $\ell$. |
| $\delta$                 | Kronecker Delta. |
| $K$                      | Maximum number of vaccination count. |
| $\mathcal{N}_{i}$        | Total population age stratification. |

</details>

### Relevant Parameters Representations

$$
\phi(t) =
\begin{cases}
\left(\sin\left(\frac{2\pi (t + \tau)}{730}\right)\right)^{1000} & \text{during seasonal vaccination periods} \\
0 & \text{otherwise}
\end{cases}
$$for

$$
\tau = 182.5 - \Delta_{t}.
$$
The function \(\eta\) determines a new immune state given the current state and the exposed strain using bitwise OR operations.

Let:
- \(x\) be the current state represented as an integer.
- \(y\) be the exposed strain represented as an integer, with \(0 \leq y \leq N - 1\), where \(N\) is the number of strains.
- \(2^y\) be the integer representation of the exposed strain.

The function performs the following steps:
1. Convert \(x\) to its binary representation.
2. Convert \(2^y\) to its binary representation.
3. Perform a bitwise OR operation between \(x\) and \(2^y\):
   \[
   \eta(x, y) = x \, | \, 2^y
   \]
4. Return the new state as an integer.

Formally, we define the new immune state function as:
\[
\eta(x, y) = x \, | \, 2^y
\]

## Example Calculations

If \(N = 2\), possible states are:
- `00` (no exposure), represented as `0`
- `01` (exposed to strain 0 only), represented as `1`
- `10` (exposed to strain 1 only), represented as `2`
- `11` (exposed to both strains), represented as `3`

1. **Initial State: 00 (0), Exposed Strain: 0**
   - Binary representation of current state (0): `00`
   - Binary representation of \(2^0\): `01`
   - Bitwise OR: `00 | 01 = 01`
   - New state: `01` (1)

2. **Initial State: 00 (0), Exposed Strain: 1**
   - Binary representation of current state (0): `00`
   - Binary representation of \(2^1\): `10`
   - Bitwise OR: `00 | 10 = 10`
   - New state: `10` (2)

3. **Initial State: 01 (1), Exposed Strain: 0**
   - Binary representation of current state (1): `01`
   - Binary representation of \(2^0\): `01`
   - Bitwise OR: `01 | 01 = 01`
   - New state: `01` (1)

4. **Initial State: 10 (2), Exposed Strain: 1**
   - Binary representation of current state (2): `10`
   - Binary representation of \(2^1\): `10`
   - Bitwise OR: `10 | 10 = 10`
   - New state: `10` (2)

5. **Initial State: 01 (1), Exposed Strain: 1**
   - Binary representation of current state (1): `01`
   - Binary representation of \(2^1\): `10`
   - Bitwise OR: `01 | 10 = 11`
   - New state: `11` (3)

6. **Initial State: 10 (2), Exposed Strain: 0**
   - Binary representation of current state (2): `10`
   - Binary representation of \(2^0\): `01`
   - Bitwise OR: `10 | 01 = 11`
   - New state: `11` (3)

7. **Initial State: 11 (3), Exposed Strain: 0**
   - Binary representation of current state (3): `11`
   - Binary representation of \(2^0\): `01`
   - Bitwise OR: `11 | 01 = 11`
   - New state: `11` (3)

8. **Initial State: 11 (3), Exposed Strain: 1**
   - Binary representation of current state (3): `11`
   - Binary representation of \(2^1\): `10`
   - Bitwise OR: `11 | 10 = 11`
   - New state: `11` (3)

Lastly, we define a mathematical representation of the Force of infection for susceptible individuals in age group \(a\) and strain \(k\) is calculated as follows:

\[
\Lambda_{a,k}(t) = \tilde{\lambda}_{a,k}(t) \cdot (1 - \text{WI}_{a,k})
\]

Where:

\[
\tilde{\lambda}_{a,k}(t) = \frac{\beta \cdot \beta_{\text{coef}}(t) \cdot \sigma(t)}{P_a} \sum_{b,i,j} C_{ab} \cdot \left( I_{b,i,j,k}(t) + \mathcal{N}(\mu_i, \sigma_i) \cdot \phi_i \cdot P_b \right)
\]

And:

\[
\text{WI}_{a,k} = \text{WIB}_{a,k} + \text{WIM}_{a,k}
\]

\[
\text{WIB}_{a,k} = \sum_{j} \text{II}_{a,j} \cdot \gamma_{jk}
\]

\[
\text{WIM}_{a,k} = (1 - \text{WIB}_{a,k}) \cdot \text{FI}_{a,k}
\]

\[
\text{FI}_{a,k} = \begin{cases}
\mathcal{M}_{\text{HI}} & \text{if previously exposed to strain } k \\
0 & \text{otherwise}
\end{cases}
\]

\[
\text{II}_{a,k} = 1 - \sum_{j} \left( 1 - \chi_{k,j} \right) \left( 1 - \nu_{k,j} \right)
\]

| Symbol               | Description                                                                                |
|----------------------|--------------------------------------------------------------------------------------------|
| \(\Lambda_{a,k}(t)\) | Effective force of infection for susceptibles in age group \(a\) and strain \(k\) at time \(t\) |
| \(\tilde{\lambda}_{a,k}(t)\) | Original force of infection (before adjusting for immunity) for age group \(a\) and strain \(k\) at time \(t\) |
| \(\beta\)            | Baseline transmission rate                                                                 |
| \(\beta_{\text{coef}}(t)\) | Time-dependent coefficient modifying \(\beta\)                                       |
| \(\sigma(t)\)        | Time-dependent seasonality coefficient                                                     |
| \(P_a\)              | Population of age group \(a\)                                                              |
| \(C_{ab}\)           | Contact matrix between age groups \(a\) and \(b\)                                          |
| \(\mathcal{N}(\mu_i, \sigma_i)\) | Normal distribution for external infection centered at \(\mu_i\) with scale \(\sigma_i\) |
| \(\phi_i\)           | Percentage of age group \(b\) for externally introduced strain \(i\)                        |
| \(\text{WI}_{a,k}\)  | Waned immunity for age group \(a\) and strain \(k\)                                         |
| \(\text{WIB}_{a,k}\) | Waned immunity baseline for age group \(a\) and strain \(k\)                                |
| \(\gamma_{jk}\)      | Waning protections matrix                                                                  |
| \(\text{WIM}_{a,k}\) | Minimum waned immunity for age group \(a\) and strain \(k\)                                 |
| \(\text{FI}_{a,k}\)  | Final immunity for age group \(a\) and strain \(k\)                                         |
| \(\mathcal{M}_{\text{HI}}\) | Minimum homologous immunity                                                        |
| \(\chi_{k,j}\)       | Cross-immunity between strain \(k\) and strain \(j\)                                        |
| \(\nu_{k,j}\)        | Vaccine efficacy against strain \(j\)                                                      |
| \(\text{II}_{a,k}\)  | Initial immunity for age group \(a\) and strain \(k\)                                       |
