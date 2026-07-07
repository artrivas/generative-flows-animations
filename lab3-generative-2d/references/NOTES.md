# Reference Notes — Lab 3: Generative 2D Visualizer

> **Verificación de fuentes**: Las ecuaciones marcadas con ⚠️ **NO fueron verificadas contra el PDF original** — fueron reproducidas desde entrenamiento y deben ser cotejadas manualmente antes del paso 2. Las ecuaciones sin marca fueron cotejadas con la fuente indicada.

---

## 1. VP SDE (Variance Preserving)

**Fuente**: Song et al. 2021, "Score-Based Generative Modeling through SDEs", ICLR 2022.  
**Sección**: §5 (Relationship to existing frameworks), también §2 (SDEs).  
**Ecuación de referencia**: Eq. (11) en el paper.  
⚠️ *No verificado contra PDF — citar número de ecuación exacto pendiente de revisión manual.*

### Forward SDE

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}
$$

donde $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$ es el noise schedule lineal, con valores típicos $\beta_{\min} = 0.1$, $\beta_{\max} = 20$ (para $t \in [0,1]$).

- **Drift**: $f(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\,\mathbf{x}$
- **Diffusion**: $g(t) = \sqrt{\beta(t)}$

**Archivo/clase de implementación**: `src/forward_process/vp_sde.py` → clase `VPSDE`

---

## 2. VE SDE (Variance Exploding)

**Fuente**: Song et al. 2021, ibid.  
**Sección**: §5 (Relationship to SMLD).  
**Ecuación de referencia**: Eq. (9) en el paper.  
⚠️ *No verificado contra PDF — número de ecuación exacto pendiente de revisión manual.*

### Forward SDE

$$
d\mathbf{x} = \sqrt{\frac{d\sigma^2(t)}{dt}}\,d\mathbf{w}
$$

con $\sigma(t) = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$, valores típicos $\sigma_{\min} = 0.01$, $\sigma_{\max} = 50$.

- **Drift**: $f(\mathbf{x}, t) = \mathbf{0}$ (no drift)
- **Diffusion**: $g(t) = \sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$

**Archivo/clase de implementación**: `src/forward_process/ve_sde.py` → clase `VESDE`

---

## 3. sub-VP SDE (Sub-Variance Preserving)

**Fuente**: Song et al. 2021, ibid.  
**Sección**: Appendix B, "Sub-VP SDE". También referenciado en §4.  
**Ecuación de referencia**: Eq. en Appendix B (número exacto pendiente de verificación).  
⚠️ *No verificado contra PDF — requiere revisión manual de Appendix B.*

### Forward SDE

$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\,\mathbf{x}\,dt + \sqrt{\beta(t)\left(1 - e^{-2\int_0^t \beta(s)\,ds}\right)}\,d\mathbf{w}
$$

El coeficiente de difusión está diseñado para que $\sigma^2_{\text{sub-VP}}(t) \leq \sigma^2_{\text{VP}}(t)$, de donde viene el nombre.

- **Drift**: $f(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\,\mathbf{x}$ (idéntico al VP)
- **Diffusion**: $g(t) = \sqrt{\beta(t)\left(1 - e^{-2\int_0^t \beta(s)\,ds}\right)}$

Para el schedule lineal, $\int_0^t \beta(s)\,ds = \beta_{\min}t + \frac{1}{2}(\beta_{\max}-\beta_{\min})t^2$.

**Archivo/clase de implementación**: `src/forward_process/subvp_sde.py` → clase `SubVPSDE`

### Kernel de transición cerrado (derivado, no citado directamente)

⚠️ **Esta fórmula NO aparece transcrita de NOTES.md/paper con número de ecuación — fue derivada analíticamente durante la implementación del paso 2 y validada numéricamente (ver `scripts/validate_forward_process.py`). Pendiente de contrastar contra Song et al. 2021, Appendix B.**

Para una SDE lineal $d\mathbf{x} = f(t)\mathbf{x}\,dt + g(t)\,d\mathbf{w}$ con $f(t) = -\frac{1}{2}\beta(t)$, la solución general es:

$$
\mathbf{x}_t = \alpha(t)\mathbf{x}_0 + \alpha(t)\int_0^t \frac{g(s)}{\alpha(s)}\,d\mathbf{w}_s, \qquad \alpha(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}
$$

Por la isometría de Itô, $\text{Var}[\mathbf{x}_t] = \alpha(t)^2\int_0^t g(s)^2/\alpha(s)^2\,ds$. Sustituyendo $g(s)^2 = \beta(s)(1-e^{-2B(s)})$ (con $B(s)=\int_0^s\beta$) e integrando (cambio de variable $u=e^{B(s)}$):

$$
\text{Var}[\mathbf{x}_t] = \left(1 - \alpha(t)^2\right)^2
$$

Es decir:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \alpha(t)\,\mathbf{x}_0,\; \left(1-\alpha(t)^2\right)^2 \mathbf{I}\right), \qquad \mu_t(\mathbf{x}_0) = \alpha(t)\mathbf{x}_0,\quad \sigma_t = 1-\alpha(t)^2
$$

Nótese $(1-\alpha(t)^2)^2 \leq 1-\alpha(t)^2 = \sigma^2_{\text{VP}}(t)$ para $\alpha(t)^2\in[0,1]$ — esta es precisamente la propiedad "sub" (varianza acotada por la del VP) que da nombre al proceso.

**Archivo/clase de implementación**: `src/forward_process/kernels.py` → función `subvp_transition_kernel`

---

## 4. Kernels de Transición Cerrados (para validación numérica)

**Fuente**: Song et al. 2021, ibid., §2 y Appendix A.  
⚠️ *No verificado contra PDF — número de ecuación exacto pendiente de revisión manual.*

Ambos kernels tienen la forma $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \mu_t(\mathbf{x}_0),\, \sigma_t^2 \mathbf{I})$.

### VP SDE — Kernel cerrado

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}\,\mathbf{x}_0,\;\left(1 - e^{-\int_0^t \beta(s)\,ds}\right)\mathbf{I}\right)
$$

Definiendo $\alpha(t) = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}$:

$$
\mu_t(\mathbf{x}_0) = \alpha(t)\,\mathbf{x}_0, \qquad \sigma_t^2 = 1 - \alpha(t)^2
$$

Para el schedule lineal: $\int_0^t \beta(s)\,ds = \beta_{\min}t + \frac{1}{2}(\beta_{\max}-\beta_{\min})t^2$.

### VE SDE — Kernel cerrado

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \mathbf{x}_0,\; [\sigma(t)^2 - \sigma(0)^2]\,\mathbf{I}\right)
$$

Con $\sigma(0) = \sigma_{\min}$, en la práctica $\sigma(0) \approx 0$:

$$
\mu_t(\mathbf{x}_0) = \mathbf{x}_0, \qquad \sigma_t^2 = \sigma(t)^2 - \sigma_{\min}^2 \approx \sigma(t)^2
$$

**Uso en validación**: El paso 2 comprobará empíricamente que las muestras de Euler–Maruyama convergen a estos kernels analíticos.

**Archivo/clase de implementación**: `src/forward_process/kernels.py` → funciones `vp_transition_kernel`, `ve_transition_kernel`

---

## 5. ε-prediction (DDPM parametrización)

**Fuente**: Ho et al. 2020, "Denoising Diffusion Probabilistic Models", NeurIPS 2020.  
**Sección**: §3.2, "Simplified training objective".  
**Ecuación de referencia**: Eq. (14) para el objetivo, Eq. (11) para la reparametrización.  
⚠️ *No verificado contra PDF — los números de ecuación son desde memoria; verificar contra el paper original.*

### Reparametrización del ruido

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

donde $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$ en la formulación discreta (análogo a $e^{-\int_0^t \beta(s)ds}$ en continuo).

### Objetivo simplificado

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\varepsilon}}\left[\left\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\right\|^2\right]
$$

### Conversión ε → score

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx -\frac{\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

**Archivo/clase de implementación**: `src/models/denoiser.py` → clase `Denoiser` (interpretado como ε-pred vía `DenoiserWrapper(param="eps")`), función `epsilon_to_score`

---

## 6. v-prediction y conversión a score

**Fuente**: Salimans & Ho 2022, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022.  
**Sección**: §2 "Continuous-time diffusion models", específicamente la definición de $v$.  
**Ecuación de referencia**: Eq. (5) y (6) en el paper.  
⚠️ *No verificado contra PDF — números de ecuación pendientes de revisión manual.*

### Definición de v

$$
\mathbf{v}_t = \sqrt{\bar{\alpha}_t}\,\boldsymbol{\varepsilon} - \sqrt{1 - \bar{\alpha}_t}\,\mathbf{x}_0
$$

Equivalentemente, dado $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\varepsilon}$:

$$
\mathbf{v}_t = \sqrt{\bar{\alpha}_t}\,\boldsymbol{\varepsilon} - \sqrt{1-\bar{\alpha}_t}\,\mathbf{x}_0
$$

### Recuperación de ε y x₀ desde v

$$
\boldsymbol{\varepsilon} = \sqrt{\bar{\alpha}_t}\,\mathbf{v}_t + \sqrt{1-\bar{\alpha}_t}\,\mathbf{x}_t
$$

$$
\mathbf{x}_0 = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\mathbf{v}_t
$$

### Conversión v → score

$$
\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) = -\frac{\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar{\alpha}_t}} = -\frac{\sqrt{\bar{\alpha}_t}\,\mathbf{v}_\theta(\mathbf{x}_t,t) + \sqrt{1-\bar{\alpha}_t}\,\mathbf{x}_t}{\sqrt{1-\bar{\alpha}_t}}
$$

**Archivo/clase de implementación**: `src/models/denoiser.py` → clase `Denoiser` (interpretado como v-pred vía `DenoiserWrapper(param="v")`), función `v_to_score`

---

## 7. Probability Flow ODE

**Fuente**: Song et al. 2021, ibid.  
**Sección**: §2, "Probability flow ODE".  
**Ecuación de referencia**: Eq. (13) en el paper.  
⚠️ *No verificado contra PDF — número de ecuación pendiente de revisión manual.*

### ODE determinista

Para cualquier SDE de la forma $d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w}$, existe la ODE de flujo de probabilidad:

$$
d\mathbf{x} = \left[f(\mathbf{x},t) - \frac{1}{2}g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\right]dt
$$

Esta ODE tiene las **mismas marginales** $p_t$ que la SDE original, pero trayectorias deterministas.

### Para VP SDE explícitamente

$$
d\mathbf{x} = \left[-\frac{1}{2}\beta(t)\,\mathbf{x} - \frac{1}{2}\beta(t)\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\right]dt
$$

**Uso en visualización**: Las trayectorias de la ODE serán comparadas con las de la SDE en los videos del paso 12.

**Archivo/clase de implementación**: `src/sampling/pf_ode.py` → función `sample_pf_ode` (usa los integradores genéricos `euler_step`/`heun_step` de `src/integrators/steps.py` y `probability_flow_drift` de `src/sampling/utils.py`)

### SDE reversa (estocástica, distinta de la Probability Flow ODE)

**Fuente**: Anderson 1982 (resultado general de reversión de SDEs); Song et al. 2021, Eq. (6).  
⚠️ *No verificado contra PDF — número de ecuación pendiente de revisión manual. Agregado en el paso 6 (revisión post-hoc) porque el código de `src/sampling/reverse_sde.py` la usa y citaba erróneamente esta sección sin que la fórmula estuviera transcrita aquí.*

Además de la ODE determinista (arriba), Song et al. 2021 muestran que el proceso reverso también puede escribirse como una SDE estocástica con las **mismas marginales** $p_t$ que el forward SDE:

$$
d\mathbf{x} = \left[f(\mathbf{x},t) - g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\right]dt + g(t)\,d\bar{\mathbf{w}}
$$

donde $\bar{\mathbf{w}}$ es un proceso de Wiener en tiempo reverso. Nótese el coeficiente $g(t)^2$ completo (no $\frac{1}{2}g(t)^2$ como en la PF-ODE) — la diferencia es exactamente el término de ruido adicional $g(t)\,d\bar{\mathbf{w}}$.

**Archivo/clase de implementación**: `src/sampling/reverse_sde.py` → función `sample_reverse_sde`; `src/sampling/utils.py` → función `reverse_sde_forward_drift`

---

## 8. Flow Matching

**Fuente**: Lipman et al. 2022, "Flow Matching for Generative Modeling", ICLR 2023.  
**Sección**: §3 "Conditional Flow Matching".  
**Ecuación de referencia**: Eq. (1) para interpolación, Eq. (6) para el campo condicional, Eq. (7) para el objetivo CFM.  
⚠️ *No verificado contra PDF — números de ecuación pendientes de revisión manual.*

### Interpolación lineal (conditional path)

$$
\mathbf{x}(t) = (1-t)\,\mathbf{x}_0 + t\,\mathbf{x}_1, \qquad t \in [0,1]
$$

donde $\mathbf{x}_0 \sim p_0$ (distribución base, típicamente $\mathcal{N}(\mathbf{0},\mathbf{I})$) y $\mathbf{x}_1 \sim p_{\text{data}}$.

### Campo de velocidad condicional objetivo

$$
u_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0
$$

Este campo es constante a lo largo de cada trayectoria lineal.

### Objetivo Conditional Flow Matching (CFM)

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\,\mathbf{x}_0,\,\mathbf{x}_1}\left[\left\|v_\theta\!\left((1-t)\mathbf{x}_0 + t\mathbf{x}_1,\; t\right) - (\mathbf{x}_1 - \mathbf{x}_0)\right\|^2\right]
$$

### Nota sobre OT-CFM

La variante OT-CFM (Liu et al. 2022 / Tong et al. 2023) emparea óptimamente $\mathbf{x}_0$ y $\mathbf{x}_1$ para reducir varianza. En este lab implementaremos primero la variante estándar (pares independientes).

**Archivo/clase de implementación**: interpolación lineal implementada inline en el loop de entrenamiento de `src/training/train_flow_matching.py` (sin clase `ForwardProcess` separada); `src/models/velocity_field.py` → clase `VelocityField`; sampling en `src/sampling/flow_matching_ode.py` → función `sample_flow_matching_ode`

---

## Mapa de implementación

| Componente | Archivo | Clase/Función |
|---|---|---|
| Interfaz base | `src/forward_process/base.py` | `ForwardProcess` |
| Schedule compartido (β lineal) | `src/forward_process/schedules.py` | `beta_linear`, `beta_integral` |
| VP SDE | `src/forward_process/vp_sde.py` | `VPSDE` |
| VE SDE | `src/forward_process/ve_sde.py` | `VESDE` |
| sub-VP SDE | `src/forward_process/subvp_sde.py` | `SubVPSDE` |
| Kernels analíticos | `src/forward_process/kernels.py` | `vp_transition_kernel`, `ve_transition_kernel`, `subvp_transition_kernel` (derivado) |
| ε-net + conversión | `src/models/denoiser.py` | `Denoiser`, `DenoiserWrapper`, `epsilon_to_score` |
| v-net + conversión | `src/models/denoiser.py` | `Denoiser`, `DenoiserWrapper`, `v_to_score` |
| Integradores genéricos (Euler/Heun/Euler-Maruyama) | `src/integrators/steps.py` | `euler_step`, `heun_step`, `euler_maruyama_step` |
| Probability Flow ODE (sampling) | `src/sampling/pf_ode.py` | `sample_pf_ode` |
| Reverse-time SDE (sampling) | `src/sampling/reverse_sde.py` | `sample_reverse_sde` |
| Flow Matching (entrenamiento) | `src/training/train_flow_matching.py` | inline, sin clase separada |
| Flow Matching (sampling) | `src/sampling/flow_matching_ode.py` | `sample_flow_matching_ode`, `load_velocity_model` |
| Velocity Net | `src/models/velocity_field.py` | `VelocityField` |

---

## Pendientes de verificación manual

Antes de comenzar el paso 2, verificar los siguientes números de ecuación contra los PDFs originales:

- [ ] Song et al. 2021 — Eq. de VP SDE (reportada como Eq. 11 o similar)
- [ ] Song et al. 2021 — Eq. de VE SDE (reportada como Eq. 9 o similar)
- [ ] Song et al. 2021 — Eq. de sub-VP SDE (Appendix B, número exacto desconocido)
- [ ] Song et al. 2021 — Kernels de transición VP/VE (Appendix A)
- [ ] Song et al. 2021 — Kernel de transición sub-VP: **derivado analíticamente en el paso 2** (ver §3), no transcrito de una ecuación del paper. Validado numéricamente vía Monte Carlo (`scripts/validate_forward_process.py`), pero pendiente de contrastar contra Appendix B por si el paper reporta esta misma forma cerrada explícitamente.
- [ ] Song et al. 2021 — Probability Flow ODE (reportada como Eq. 13)
- [ ] Ho et al. 2020 — Objetivo ε-prediction (Eq. 14) y reparametrización (Eq. 11)
- [ ] Salimans & Ho 2022 — Definición v-prediction (Eq. 5-6)
- [ ] Lipman et al. 2022 — Interpolación y campo CFM (Eq. 1, 6, 7)
